/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/xla/service/hlo_cpu_instrumentation.h"

#include <chrono>
#include <ctime>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/service/custom_call_status.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
namespace xla {

namespace cpu_instr {
int call_index = 0;
} // namespace cpu_instr

void CpuInstrCallback(void *output, void **inputs,
                      XlaCustomCallStatus *status) {
  // const float* index = reinterpret_cast<const float*>(inputs[0]);

  auto ts = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count();

  std::cout << "Instr call at:" << xla::cpu_instr::call_index++ << ' ';
  std::cout << ts << std::endl;
}

XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cpu_instrumentation_callback",
                                         &CpuInstrCallback, "Host");

/*static*/ StatusOr<bool>
HloCpuInstr::RunOnComputation(HloComputation *computation) {
  bool changed = true;

  return changed;
}
StatusOr<bool> HloCpuInstr::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  bool changed = true;

  auto main_computation = module->entry_computation();

  // auto instr = Cast<HloCustomCallInstruction>(
  //     main_computation->AddInstruction(HloInstruction::CreateCustomCall(
  //         ShapeUtil::MakeShape(F32, {}),
  //         /*operands=*/{},
  //         /*custom_call_target=*/"cpu_instrumentation_callback")));
  // instr->set_custom_call_has_side_effect(true);
  // return changed;

  std::vector<xla::HloInstruction *> instructions;
  for (xla::HloInstruction *instruction : main_computation->instructions()) {
    instructions.push_back(instruction);
  }
  int instr_index = 0;
  for (xla::HloInstruction *instruction : instructions) {
    std::cout << instruction->name() << std::endl;
    // xla::Array<float> index_arr = {float(instr_index++)};

    // auto index = Cast<HloInstruction>(
    //     main_computation->AddInstruction(HloInstruction::CreateParameter(
    //         index_arr, ShapeUtil::MakeShape(F32, {}),
    //         "instrumentation_index")));
    // auto index = Cast<HloInstruction>(main_computation->AddInstruction(
    //     HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>())));

    auto instr = Cast<HloCustomCallInstruction>(
        main_computation->AddInstruction(HloInstruction::CreateCustomCall(
            ShapeUtil::MakeShape(F32, {}),
            /*operands=*/{},
            /*custom_call_target=*/"cpu_instrumentation_callback")));
    instr->set_custom_call_has_side_effect(true);
    TF_RETURN_IF_ERROR(instruction->AddControlDependencyTo(instr));
    for (xla::HloInstruction *user : instruction->users()) {
      TF_RETURN_IF_ERROR(instr->AddControlDependencyTo(user));
    }
  }

  return changed;
}

} // namespace xla
