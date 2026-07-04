// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// In-memory staging gate for the plan-level sandbox create intent (#6226).
//
// The machine sandbox phase (`createSandboxPhase`) opens a draft with the raw
// flow-context inputs before the sandbox handler runs; messaging preflight
// (`prepareSandboxMessagingPreflight`) completes the draft into a full
// `SandboxCreateIntent` — the serializable plan-level intent from
// `sandbox-create-intent-types.ts`, NOT the machine-level
// `SandboxCreateIntent { recreate; toolDisclosure }` in `./types.ts` — and
// validates its credential bindings before the destructive recreate sequence
// in `createSandboxWithBaseImageResolution` can run. The validated intent
// stays in this module's memory only; it is never persisted or emitted as a
// machine or public event. Checkpointing and consumption belong to #6228.
//
// Boundary coverage: preflight completion runs before every destructive
// effect inside `createSandboxWithBaseImageResolution` (recreate backup,
// pre-delete cleanup, `sandbox delete`, docker rmi, registry removal). It
// does NOT cover the earlier machine resume arms: `applySandboxResumeDecision`
// can remove the registry entry, and its repair-and-recreate arm deletes the
// recorded sandbox via `repairRecordedSandbox`, before preflight validation
// runs. Gating those arms requires the frozen sandbox handler and is deferred
// until #5595/#6253 land.
//
// Every production `deps.createSandbox` caller — including the prepared
// dcode/image rebuild handoff bound via
// `prepared-dcode-rebuild.ts#bindCreateSandbox` — executes under
// `createSandboxPhase`, so the draft is open and completion is live for all
// of them. The only no-draft caller is the module-level `createSandbox`
// export of `onboard.ts` (no production importer; test harnesses only), for
// which completion is deliberately a no-op.
//
// The draft stages `hermesToolGateways` raw from flow context. The effective
// hermes gateway list (web-search interplay) is computed at apply time inside
// the frozen sandbox handler; unify once #5595 lands.

import path from "node:path";

import { isLinuxDockerDriverGatewayEnabled } from "./docker-driver-platform";
import type { MessagingTokenDef, NamedMessagingChannel } from "./messaging-prep";
import {
  getPrimarySandboxCreateCredentialEnvKeys,
  readPolicyTierEnv,
  resolveSandboxCreateIntent,
  resolveSandboxCreateMessagingProviderRequests,
  type SandboxCreateIntent,
  validateSandboxCreateIntentCredentialBindings,
} from "./sandbox-create-plan";
import { buildSandboxGpuCreateArgs, type SandboxGpuCreateConfig } from "./sandbox-gpu-create";

export type SandboxCreateIntentDraft = {
  /** Opaque flow-context values; narrowed only at completion time. */
  sandboxGpuConfig: unknown;
  hermesToolGateways: readonly string[];
  agent: unknown;
};

type ResolveDockerGpuSandboxCreatePlan =
  typeof import("./docker-gpu-sandbox-create").resolveDockerGpuSandboxCreatePlan;

export type SandboxCreateIntentGateDeps = {
  resolveDockerGpuSandboxCreatePlan?: ResolveDockerGpuSandboxCreatePlan;
  dockerDriverGateway?: boolean;
  listExtraProviders?: () => readonly string[];
  getRepoRoot?: () => string;
};

export type CompleteSandboxCreateIntentDraftInput = {
  input: {
    sandboxName: string;
    channels: readonly NamedMessagingChannel[];
    enabledChannels: readonly string[] | null;
  };
  result: {
    messagingTokenDefs: MessagingTokenDef[];
    disabledChannelNames: ReadonlySet<string>;
    reusableMessagingChannels: readonly string[];
    reusableMessagingProviders: readonly string[];
  };
  getMessagingChannelForEnvKey(envKey: string): string | null;
  deps?: SandboxCreateIntentGateDeps;
};

// Module-scoped single slot: the only collision-free bridge across the frozen
// onboard frames until explicit plumbing can land post-#5595. Opened and
// cleared by `createSandboxPhase`, so the slot never outlives one phase run.
let draft: SandboxCreateIntentDraft | null = null;
let stagedIntent: SandboxCreateIntent | null = null;

export function openSandboxCreateIntentDraft(next: SandboxCreateIntentDraft): void {
  draft = next;
  stagedIntent = null;
}

/**
 * Read hook reserved for #6228's checkpoint/consumption work; until then only
 * tests observe staging through it.
 */
export function peekStagedSandboxCreateIntent(): SandboxCreateIntent | null {
  return stagedIntent;
}

export function clearSandboxCreateIntentGate(): void {
  draft = null;
  stagedIntent = null;
}

// Lazy requires mirror `sandbox-create-plan.ts`: keep modules that
// transitively reach `runner.ts` out of this module's static graph so vitest
// source resolution keeps working for every importer.
function getDockerGpuSandboxCreatePlan(
  ...args: Parameters<ResolveDockerGpuSandboxCreatePlan>
): ReturnType<ResolveDockerGpuSandboxCreatePlan> {
  const { resolveDockerGpuSandboxCreatePlan } =
    require("./docker-gpu-sandbox-create") as typeof import("./docker-gpu-sandbox-create");
  return resolveDockerGpuSandboxCreatePlan(...args);
}

function listRegistryExtraProviders(): readonly string[] {
  const { listExtraProviders } = require("../state/registry") as typeof import("../state/registry");
  return listExtraProviders();
}

function getDefaultRepoRoot(): string {
  const { ROOT } = require("../runner") as typeof import("../runner");
  return ROOT;
}

/**
 * Complete the open draft into a fully resolved `SandboxCreateIntent`,
 * validate its credential bindings against the live messaging token defs, and
 * stage the validated intent in memory. Returns null without any effect when
 * no draft is open. Validation failures (the `bindMessagingTokenDefs` drift
 * errors) propagate to the caller — fail closed, nothing staged.
 */
export function completeSandboxCreateIntentDraft({
  input,
  result,
  getMessagingChannelForEnvKey,
  deps = {},
}: CompleteSandboxCreateIntentDraftInput): SandboxCreateIntent | null {
  if (!draft) return null;
  // Narrow casts follow the frozen sandbox handler's precedent: flow context
  // stores these fields opaquely, and completion reads only the create-time
  // fields the fresh resolution in `createSandboxWithBaseImageResolution`
  // reads.
  const agent = draft.agent as { name?: string; policyAdditionsPath?: string | null } | null;
  const sandboxGpuConfig = draft.sandboxGpuConfig as SandboxGpuCreateConfig;

  const { useDockerGpuPatch, logMessage: sandboxGpuLogMessage } = (
    deps.resolveDockerGpuSandboxCreatePlan ?? getDockerGpuSandboxCreatePlan
  )(sandboxGpuConfig, {
    dockerDriverGateway: deps.dockerDriverGateway ?? isLinuxDockerDriverGatewayEnabled(),
  });
  const gpuCreateArgs = buildSandboxGpuCreateArgs(sandboxGpuConfig, {
    suppressGpuFlag: useDockerGpuPatch,
  });
  const defaultPolicyPath = path.join(
    (deps.getRepoRoot ?? getDefaultRepoRoot)(),
    "nemoclaw-blueprint",
    "policies",
    "openclaw-sandbox.yaml",
  );

  const intent = resolveSandboxCreateIntent({
    // Inline of `agentOnboard.getAgentPolicyPath` over the narrow agent shape,
    // matching the fresh `basePolicyPath` derivation in onboard.ts.
    basePolicyPath: (agent && (agent.policyAdditionsPath || null)) || defaultPolicyPath,
    sandboxName: input.sandboxName,
    channels: [...input.channels],
    enabledChannels: input.enabledChannels && [...input.enabledChannels],
    disabledChannelNames: result.disabledChannelNames,
    messagingProviderRequests: resolveSandboxCreateMessagingProviderRequests(
      result.messagingTokenDefs,
      getMessagingChannelForEnvKey,
    ),
    primaryMessagingCredentialEnvKeys: getPrimarySandboxCreateCredentialEnvKeys(),
    reusableMessagingChannels: result.reusableMessagingChannels,
    reusableMessagingProviders: result.reusableMessagingProviders,
    extraProviders: (deps.listExtraProviders ?? listRegistryExtraProviders)(),
    hermesToolGateways: [...draft.hermesToolGateways],
    sandboxGpuConfig,
    gpuCreateArgs,
    useDockerGpuPatch,
    sandboxGpuLogMessage,
    // Raw agent name (undefined for the default agent), matching the fresh
    // resolution's `agent?.name` — NOT preflight's `agent?.name ?? "openclaw"`.
    agentName: agent?.name,
    policyTier: readPolicyTierEnv(),
  });
  validateSandboxCreateIntentCredentialBindings(intent, result.messagingTokenDefs);
  stagedIntent = intent;
  return intent;
}
