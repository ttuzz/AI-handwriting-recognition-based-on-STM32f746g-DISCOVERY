/**
 ******************************************************************************
 * @file    aiValidation.h
 * @author  MCD/AIS Team
 * @brief   AI Validation application
 ******************************************************************************
 * @attention
 *
 * <h2><center>&copy; Copyright (c) 2019,2021 STMicroelectronics.
 * All rights reserved.</center></h2>
 *
 * This software is licensed under terms that can be found in the LICENSE file in
 * the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */

#ifndef __AI_VALIDATION_H__
#define __AI_VALIDATION_H__

#include <stdint.h>
#include "aiPbMgr.h"
#include "app_x-cube-ai.h"
#if defined(AI_MNETWORK_IN_1_SIZE_BYTES)
AI_ALIGNED(4)
static ai_u8 in_data[AI_MNETWORK_IN_1_SIZE_BYTES];

AI_ALIGNED(4)
static ai_u8 out_data[AI_MNETWORK_OUT_1_SIZE_BYTES];
#else
static ai_float in_data[AI_MNETWORK_IN_1_SIZE];
static ai_float out_data[AI_MNETWORK_OUT_1_SIZE];
#endif
#ifdef __cplusplus
extern "C" {
#endif

int aiValidationInit(void);
int aiValidationProcess(void);
void aiValidationDeInit(void);

#ifdef __cplusplus
}
#endif

#endif /* __AI_VALIDATION_H__ */
