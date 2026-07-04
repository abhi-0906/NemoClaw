// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Wiring contract between messaging preflight and the #6226 create-intent
// gate. Kept out of sandbox-messaging-preflight.test.ts so the legacy suite
// stays focused on conflict/credential guards.

import { afterEach, describe, expect, it, vi } from "vitest";
import { listChannels } from "../sandbox/channels";
import {
  clearSandboxCreateIntentGate,
  peekStagedSandboxCreateIntent,
} from "./sandbox-create-intent-gate";
import {
  prepareSandboxMessagingPreflight,
  type SandboxMessagingPreflightDeps,
} from "./sandbox-messaging-preflight";

class ExitError extends Error {
  constructor(readonly code: number) {
    super(`exit ${code}`);
  }
}

function createResult(overrides = {}) {
  return {
    disabledChannelNames: new Set<string>(),
    messagingTokenDefs: [],
    extraPlaceholderKeys: [],
    hasMessagingTokens: false,
    reusableMessagingProviders: [],
    reusableMessagingChannels: [],
    missingWebSearchCredentialEnv: null,
    ...overrides,
  };
}

function createDeps(
  overrides: Partial<SandboxMessagingPreflightDeps> = {},
): SandboxMessagingPreflightDeps {
  return {
    readMessagingPlanFromEnv: vi.fn(() => null),
    resolveDisabledChannels: vi.fn(() => []),
    gatewayName: "nemoclaw",
    registry: {
      listSandboxes: vi.fn(() => ({ sandboxes: [] })),
    },
    providerExistsInGateway: vi.fn(() => false),
    isNonInteractive: vi.fn(() => false),
    promptYesNoOrDefault: vi.fn(async () => true),
    cliName: vi.fn(() => "nemoclaw"),
    log: vi.fn(),
    error: vi.fn(),
    exitProcess: vi.fn((code: number) => {
      throw new ExitError(code);
    }) as (code: number) => never,
    getValidatedMessagingTokenByEnvKey: vi.fn(() => null),
    getCredential: vi.fn(() => null),
    normalizeCredentialValue: vi.fn((value: unknown) =>
      typeof value === "string" ? value.trim() : "",
    ),
    registerExtraPlaceholderProviders: vi.fn(() => []),
    getMessagingChannelForEnvKey: vi.fn(() => null),
    prepareCreateSandboxMessaging: vi.fn((input) =>
      createResult({ disabledChannelNames: new Set(input.disabledChannels) }),
    ),
    ...overrides,
  };
}

const baseInput = {
  sandboxName: "demo",
  channels: listChannels(),
  enabledChannels: ["slack"],
  webSearchConfig: null,
  env: {},
};

afterEach(() => {
  clearSandboxCreateIntentGate();
});

describe("prepareSandboxMessagingPreflight create-intent gate wiring (#6226)", () => {
  it("invokes intent-draft completion with preflight outputs after the guards pass", async () => {
    const completeSandboxCreateIntentDraft = vi.fn(() => null);
    const messagingTokenDefs = [
      { name: "demo-telegram-bridge", envKey: "TELEGRAM_BOT_TOKEN", token: "secret" },
    ];
    const deps = createDeps({
      completeSandboxCreateIntentDraft,
      prepareCreateSandboxMessaging: vi.fn(() => createResult({ messagingTokenDefs })),
    });

    await prepareSandboxMessagingPreflight(baseInput, deps);

    expect(completeSandboxCreateIntentDraft).toHaveBeenCalledTimes(1);
    expect(completeSandboxCreateIntentDraft).toHaveBeenCalledWith({
      input: baseInput,
      result: expect.objectContaining({
        messagingTokenDefs,
        disabledChannelNames: new Set<string>(),
      }),
      getMessagingChannelForEnvKey: deps.getMessagingChannelForEnvKey,
    });
  });

  it("skips completion when the web-search credential guard aborts", async () => {
    const completeSandboxCreateIntentDraft = vi.fn(() => null);
    const deps = createDeps({
      completeSandboxCreateIntentDraft,
      prepareCreateSandboxMessaging: vi.fn(() =>
        createResult({ missingWebSearchCredentialEnv: "BRAVE_API_KEY" }),
      ),
    });

    await expect(prepareSandboxMessagingPreflight(baseInput, deps)).rejects.toMatchObject({
      code: 1,
    });
    expect(completeSandboxCreateIntentDraft).not.toHaveBeenCalled();
  });

  it("propagates completion validation failures before returning a result", async () => {
    const driftError = new Error(
      "Cannot materialize sandbox create intent; credential availability changed for provider 'demo-telegram-bridge'.",
    );
    const deps = createDeps({
      completeSandboxCreateIntentDraft: vi.fn(() => {
        throw driftError;
      }),
    });

    await expect(prepareSandboxMessagingPreflight(baseInput, deps)).rejects.toBe(driftError);
  });

  it("resolves normally through the real gate when no draft is open", async () => {
    const deps = createDeps();

    const result = await prepareSandboxMessagingPreflight(baseInput, deps);

    expect(result.disabledChannels).toEqual([]);
    expect(peekStagedSandboxCreateIntent()).toBeNull();
  });
});
