// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  clearSandboxCreateIntentGate,
  openSandboxCreateIntentDraft,
} from "../../sandbox-create-intent-gate";
import type {
  OnboardFlowContext,
  OnboardFlowPhaseResult,
  ProviderModelSelectedOnboardFlowContext,
  ProviderSelectedOnboardFlowContext,
  SandboxCreatedOnboardFlowContext,
} from "../flow-context";
import { assertProviderSelectedContext, onboardFlowPhaseResult } from "../flow-context";
import type { OnboardSequencePhase } from "../sequence-runner";

type ProviderInferencePhaseHandler<Context extends OnboardFlowContext> = (
  context: Context,
) => Promise<{
  context: ProviderModelSelectedOnboardFlowContext<Context>;
  result: OnboardFlowPhaseResult<Context>["result"];
}>;

type SandboxPhaseHandler<Context extends OnboardFlowContext> = (
  context: ProviderSelectedOnboardFlowContext<Context>,
) => Promise<{
  context: SandboxCreatedOnboardFlowContext<Context>;
  result: OnboardFlowPhaseResult<Context>["result"];
}>;

export function createProviderInferencePhase<Context extends OnboardFlowContext>(
  runProviderInference: ProviderInferencePhaseHandler<Context>,
): OnboardSequencePhase<Context> {
  return {
    state: "provider_selection",
    async run(context) {
      const result = await runProviderInference(context);
      return onboardFlowPhaseResult(result.context, result.result);
    },
  };
}

export function createSandboxPhase<Context extends OnboardFlowContext>(
  runSandbox: SandboxPhaseHandler<Context>,
): OnboardSequencePhase<Context> {
  return {
    state: "sandbox",
    async run(context) {
      assertProviderSelectedContext(context, "sandbox setup");
      // #6226: stage the raw create-intent inputs before the sandbox handler
      // runs; messaging preflight completes and validates the draft before
      // the destructive sequence in createSandboxWithBaseImageResolution.
      // The handler's earlier resume arms (registry removal, repair delete)
      // are not gated yet — see the gate module header. Cleared on success
      // and on throw so the in-memory slot never outlives one phase run.
      openSandboxCreateIntentDraft({
        sandboxGpuConfig: context.sandboxGpuConfig,
        hermesToolGateways: context.hermesToolGateways,
        agent: context.agent,
      });
      try {
        const result = await runSandbox(context);
        return onboardFlowPhaseResult(result.context, result.result);
      } finally {
        clearSandboxCreateIntentGate();
      }
    },
  };
}
