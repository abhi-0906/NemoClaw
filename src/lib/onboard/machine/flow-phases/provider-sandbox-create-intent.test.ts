// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// #6226 wiring for createSandboxPhase: the create-intent draft opens before
// the sandbox handler runs and is cleared on success and on throw. Kept out
// of provider-sandbox.test.ts so the legacy suite stays focused on the
// context handoff contract.

import { afterEach, describe, expect, it, vi } from "vitest";

import { createSession } from "../../../state/onboard-session";
import {
  type CompleteSandboxCreateIntentDraftInput,
  clearSandboxCreateIntentGate,
  completeSandboxCreateIntentDraft,
  peekStagedSandboxCreateIntent,
} from "../../sandbox-create-intent-gate";
import type { SandboxCreateIntent } from "../../sandbox-create-plan";
import type { SandboxGpuCreateConfig } from "../../sandbox-gpu-create";
import type { OnboardFlowContext } from "../flow-context";
import { branchTo } from "../result";
import { createSandboxPhase } from "./provider-sandbox";

type TestAgent = { name: string } | null;
type TestContext = OnboardFlowContext<TestAgent, null, SandboxGpuCreateConfig>;

function context(patch: Partial<TestContext> = {}): TestContext {
  return {
    resume: false,
    fresh: false,
    session: createSession(),
    agent: null,
    recordedSandboxName: null,
    requestedSandboxName: null,
    sandboxName: null,
    fromDockerfile: null,
    model: "model",
    provider: "nvidia-prod",
    endpointUrl: null,
    credentialEnv: null,
    hermesAuthMethod: null,
    hermesToolGateways: [],
    preferredInferenceApi: null,
    compatibleEndpointReasoning: null,
    nimContainer: null,
    webSearchConfig: null,
    webSearchSupported: false,
    selectedMessagingChannels: [],
    gpu: null,
    sandboxGpuConfig: { sandboxGpuEnabled: false },
    gpuPassthrough: false,
    ...patch,
  };
}

function sandboxCreatedContext(current: TestContext) {
  return {
    ...current,
    session: createSession(),
    sandboxName: "my-assistant",
    model: "model",
    provider: "nvidia-prod",
    webSearchConfig: null,
    selectedMessagingChannels: [],
    webSearchSupported: false,
  };
}

// Completion doubles as the draft-open probe: it returns an intent only when
// a draft is open, and null otherwise.
function stubCompletionInput(): CompleteSandboxCreateIntentDraftInput {
  return {
    input: { sandboxName: "my-assistant", channels: [], enabledChannels: [] },
    result: {
      messagingTokenDefs: [],
      disabledChannelNames: new Set<string>(),
      reusableMessagingChannels: [],
      reusableMessagingProviders: [],
    },
    getMessagingChannelForEnvKey: () => null,
    deps: {
      resolveDockerGpuSandboxCreatePlan: () => ({ useDockerGpuPatch: false, logMessage: null }),
      dockerDriverGateway: false,
      listExtraProviders: () => [],
      getRepoRoot: () => "/repo",
    },
  };
}

afterEach(() => {
  clearSandboxCreateIntentGate();
  vi.unstubAllEnvs();
});

describe("createSandboxPhase create-intent draft (#6226)", () => {
  it("opens the draft before the handler runs and clears it on success", async () => {
    const draftStates: boolean[] = [];
    const runSandbox = vi.fn(async (current: TestContext) => {
      draftStates.push(completeSandboxCreateIntentDraft(stubCompletionInput()) !== null);
      return { context: sandboxCreatedContext(current), result: branchTo("openclaw") };
    });
    const phase = createSandboxPhase<TestContext>(runSandbox);

    expect(completeSandboxCreateIntentDraft(stubCompletionInput())).toBeNull();
    await phase.run(context());

    expect(draftStates).toEqual([true]);
    expect(completeSandboxCreateIntentDraft(stubCompletionInput())).toBeNull();
    expect(peekStagedSandboxCreateIntent()).toBeNull();
  });

  it("clears the draft when the handler throws", async () => {
    const phase = createSandboxPhase<TestContext>(
      vi.fn(async () => {
        throw new Error("sandbox handler failed");
      }),
    );

    await expect(phase.run(context())).rejects.toThrow("sandbox handler failed");

    expect(completeSandboxCreateIntentDraft(stubCompletionInput())).toBeNull();
    expect(peekStagedSandboxCreateIntent()).toBeNull();
  });

  it("opens no draft when the context assertion rejects execution", async () => {
    const runSandbox = vi.fn(async (current: TestContext) => ({
      context: sandboxCreatedContext(current),
      result: branchTo("openclaw"),
    }));
    const phase = createSandboxPhase<TestContext>(runSandbox);

    await expect(phase.run(context({ provider: null }))).rejects.toThrow(
      /Onboarding state is incomplete before sandbox setup\./,
    );

    expect(runSandbox).not.toHaveBeenCalled();
    expect(completeSandboxCreateIntentDraft(stubCompletionInput())).toBeNull();
  });

  it("stages flow-context values that completion resolves into the intent", async () => {
    vi.stubEnv("NEMOCLAW_NON_INTERACTIVE", "0");
    const captured: (SandboxCreateIntent | null)[] = [];
    const runSandbox = vi.fn(async (current: TestContext) => {
      captured.push(
        completeSandboxCreateIntentDraft({
          input: { sandboxName: "my-assistant", channels: [], enabledChannels: [] },
          result: {
            messagingTokenDefs: [],
            disabledChannelNames: new Set<string>(),
            reusableMessagingChannels: [],
            reusableMessagingProviders: [],
          },
          getMessagingChannelForEnvKey: () => null,
          deps: {
            resolveDockerGpuSandboxCreatePlan: vi.fn(() => ({
              useDockerGpuPatch: false,
              logMessage: null,
            })),
            dockerDriverGateway: false,
            listExtraProviders: () => [],
            getRepoRoot: () => "/repo",
          },
        }),
        peekStagedSandboxCreateIntent(),
      );
      return { context: sandboxCreatedContext(current), result: branchTo("openclaw") };
    });
    const phase = createSandboxPhase<TestContext>(runSandbox);

    await phase.run(context({ agent: { name: "hermes" }, hermesToolGateways: ["github"] }));

    const [intentInsideHandler, stagedInsideHandler] = captured;
    expect(intentInsideHandler).toMatchObject({
      sandboxName: "my-assistant",
      hermesToolGateways: ["github"],
    });
    expect(intentInsideHandler?.policy.options.agentName).toBe("hermes");
    expect(intentInsideHandler?.policy.options.directGpu).toBe(false);
    expect(stagedInsideHandler).toBe(intentInsideHandler);
    expect(peekStagedSandboxCreateIntent()).toBeNull();
  });
});
