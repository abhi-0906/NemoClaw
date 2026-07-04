// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { MessagingTokenDef } from "./messaging-prep";
import {
  type CompleteSandboxCreateIntentDraftInput,
  clearSandboxCreateIntentGate,
  completeSandboxCreateIntentDraft,
  openSandboxCreateIntentDraft,
  peekStagedSandboxCreateIntent,
  type SandboxCreateIntentDraft,
} from "./sandbox-create-intent-gate";
import {
  getPrimarySandboxCreateCredentialEnvKeys,
  resolveSandboxCreateIntent,
  resolveSandboxCreateMessagingProviderRequests,
} from "./sandbox-create-plan";
import type { SandboxGpuCreateConfig } from "./sandbox-gpu-create";

const sandboxGpuConfig: SandboxGpuCreateConfig = {
  sandboxGpuEnabled: true,
  sandboxGpuDevice: "nvidia.com/gpu=0",
};

const channels = [
  {
    name: "telegram",
    envKey: "TELEGRAM_BOT_TOKEN",
    label: "Telegram",
    description: "Telegram",
    help: "Telegram",
  },
  {
    name: "slack",
    envKey: "SLACK_BOT_TOKEN",
    appTokenEnvKey: "SLACK_APP_TOKEN",
    label: "Slack",
    description: "Slack",
    help: "Slack",
  },
];

const telegramTokenDef: MessagingTokenDef = {
  name: "sandbox-telegram-bridge",
  envKey: "TELEGRAM_BOT_TOKEN",
  token: "telegram-super-secret",
};

const channelForEnvKey = (envKey: string) =>
  envKey === "TELEGRAM_BOT_TOKEN" ? "telegram" : envKey === "SLACK_BOT_TOKEN" ? "slack" : null;

function openDraft(patch: Partial<SandboxCreateIntentDraft> = {}): void {
  openSandboxCreateIntentDraft({
    sandboxGpuConfig,
    hermesToolGateways: ["github"],
    agent: { name: "hermes", policyAdditionsPath: "/agents/hermes/policy.yaml" },
    ...patch,
  });
}

function completionInput(
  overrides: Partial<CompleteSandboxCreateIntentDraftInput> = {},
): CompleteSandboxCreateIntentDraftInput {
  return {
    input: {
      sandboxName: "sandbox",
      channels,
      enabledChannels: ["telegram"],
    },
    result: {
      messagingTokenDefs: [telegramTokenDef],
      disabledChannelNames: new Set<string>(),
      reusableMessagingChannels: [],
      reusableMessagingProviders: [],
    },
    getMessagingChannelForEnvKey: channelForEnvKey,
    deps: {
      resolveDockerGpuSandboxCreatePlan: vi.fn(() => ({
        useDockerGpuPatch: false,
        logMessage: null,
      })),
      dockerDriverGateway: true,
      listExtraProviders: () => ["custom-provider"],
      getRepoRoot: () => "/repo",
    },
    ...overrides,
  };
}

afterEach(() => {
  clearSandboxCreateIntentGate();
  vi.unstubAllEnvs();
});

describe("sandbox create intent gate", () => {
  it("opens, completes, and clears the draft lifecycle", () => {
    expect(peekStagedSandboxCreateIntent()).toBeNull();
    expect(completeSandboxCreateIntentDraft(completionInput())).toBeNull();

    openDraft();
    expect(peekStagedSandboxCreateIntent()).toBeNull();
    expect(completeSandboxCreateIntentDraft(completionInput())).not.toBeNull();

    clearSandboxCreateIntentGate();
    expect(peekStagedSandboxCreateIntent()).toBeNull();
    expect(completeSandboxCreateIntentDraft(completionInput())).toBeNull();
  });

  it("completes as a no-op without touching any dependency when no draft is open", () => {
    const resolveDockerGpuSandboxCreatePlan = vi.fn(() => ({
      useDockerGpuPatch: false,
      logMessage: null,
    }));
    const listExtraProviders = vi.fn(() => []);
    const getRepoRoot = vi.fn(() => "/repo");

    const staged = completeSandboxCreateIntentDraft(
      completionInput({
        deps: { resolveDockerGpuSandboxCreatePlan, listExtraProviders, getRepoRoot },
      }),
    );

    expect(staged).toBeNull();
    expect(peekStagedSandboxCreateIntent()).toBeNull();
    expect(resolveDockerGpuSandboxCreatePlan).not.toHaveBeenCalled();
    expect(listExtraProviders).not.toHaveBeenCalled();
    expect(getRepoRoot).not.toHaveBeenCalled();
  });

  it("resolves, validates, and stages an intent matching fresh-resolution semantics", () => {
    vi.stubEnv("NEMOCLAW_NON_INTERACTIVE", "1");
    vi.stubEnv("NEMOCLAW_POLICY_TIER", "balanced");
    openDraft();

    const staged = completeSandboxCreateIntentDraft(completionInput());

    const expected = resolveSandboxCreateIntent({
      basePolicyPath: "/agents/hermes/policy.yaml",
      sandboxName: "sandbox",
      channels,
      enabledChannels: ["telegram"],
      disabledChannelNames: new Set(),
      messagingProviderRequests: resolveSandboxCreateMessagingProviderRequests(
        [telegramTokenDef],
        channelForEnvKey,
      ),
      primaryMessagingCredentialEnvKeys: getPrimarySandboxCreateCredentialEnvKeys(),
      reusableMessagingChannels: [],
      reusableMessagingProviders: [],
      extraProviders: ["custom-provider"],
      hermesToolGateways: ["github"],
      sandboxGpuConfig,
      gpuCreateArgs: ["--gpu", "--gpu-device", "nvidia.com/gpu=0"],
      useDockerGpuPatch: false,
      sandboxGpuLogMessage: null,
      agentName: "hermes",
      policyTier: "balanced",
    });
    expect(staged).toEqual(expected);
    expect(peekStagedSandboxCreateIntent()).toBe(staged);
    expect(JSON.parse(JSON.stringify(staged))).toEqual(staged);
    expect(JSON.stringify(staged)).not.toContain("telegram-super-secret");
  });

  it("omits agentName and uses the default policy path for the default agent", () => {
    vi.stubEnv("NEMOCLAW_NON_INTERACTIVE", "0");
    openDraft({ agent: null, hermesToolGateways: [] });

    const staged = completeSandboxCreateIntentDraft(completionInput());

    expect(staged?.policy.basePolicyPath).toBe(
      path.join("/repo", "nemoclaw-blueprint", "policies", "openclaw-sandbox.yaml"),
    );
    expect(staged?.policy.options).not.toHaveProperty("agentName");
    expect(staged?.policy.options.policyTier).toBeNull();
  });

  it("suppresses the GPU flag when the docker GPU patch plan is selected", () => {
    openDraft();
    const resolveDockerGpuSandboxCreatePlan = vi.fn(() => ({
      useDockerGpuPatch: true,
      logMessage: "gpu note",
    }));

    const staged = completeSandboxCreateIntentDraft(
      completionInput({
        deps: {
          resolveDockerGpuSandboxCreatePlan,
          dockerDriverGateway: true,
          listExtraProviders: () => [],
          getRepoRoot: () => "/repo",
        },
      }),
    );

    expect(resolveDockerGpuSandboxCreatePlan).toHaveBeenCalledWith(sandboxGpuConfig, {
      dockerDriverGateway: true,
    });
    expect(staged?.gpuCreateArgs).toEqual([]);
    expect(staged?.useDockerGpuPatch).toBe(true);
    expect(staged?.sandboxGpuLogMessage).toBe("gpu note");
    expect(staged?.policy.options.dockerGpuPatch).toBe(true);
  });

  it("throws the existing binding drift error and stages nothing", () => {
    openDraft();
    const conflictingTokenDefs: MessagingTokenDef[] = [
      { name: "sandbox-telegram-bridge", envKey: "TELEGRAM_BOT_TOKEN", token: "secret" },
      { name: "sandbox-telegram-bridge", envKey: "TELEGRAM_BOT_TOKEN", token: null },
    ];

    expect(() =>
      completeSandboxCreateIntentDraft(
        completionInput({
          result: {
            messagingTokenDefs: conflictingTokenDefs,
            disabledChannelNames: new Set<string>(),
            reusableMessagingChannels: [],
            reusableMessagingProviders: [],
          },
        }),
      ),
    ).toThrow(
      "Cannot materialize sandbox create intent; credential availability changed for provider 'sandbox-telegram-bridge'.",
    );
    expect(peekStagedSandboxCreateIntent()).toBeNull();
    // The draft survives a failed completion: a corrected retry succeeds.
    expect(completeSandboxCreateIntentDraft(completionInput())).not.toBeNull();
  });

  it("discards a previously staged intent when a new draft opens", () => {
    openDraft();
    completeSandboxCreateIntentDraft(completionInput());
    expect(peekStagedSandboxCreateIntent()).not.toBeNull();

    openDraft({ hermesToolGateways: [] });

    expect(peekStagedSandboxCreateIntent()).toBeNull();
  });
});
