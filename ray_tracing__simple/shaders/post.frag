/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#version 450

layout(location = 0) in vec2 outUV;
layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform sampler2D noisyTxt;   // color (rgba32f)
layout(set = 0, binding = 1) uniform sampler2D normalTxt;  // world normal (rgba16f, xyz used)
layout(set = 0, binding = 2) uniform sampler2D uDepth;     // linear depth (r32f)
layout(set = 0, binding = 3) uniform sampler2D uMotion;
layout(set = 0, binding = 4) uniform sampler2D uNoisyShadow;   
layout(set = 0, binding = 5) uniform sampler2D uDenoisedShadow;

layout(push_constant) uniform PC {
  float aspect;
  int   mode;   // 0=color, 1=normals, 2=depth
} pc;

vec3 visualizeNormal(vec3 N)
{
  // In case the buffer isn’t perfectly normalized
  float len = length(N);
  if (len > 1e-6) N /= len;
  return 0.5 * N + 0.5;  // map [-1,1] -> [0,1]
}

void main()
{
  // We'll fetch normals/depth by pixel to avoid filtering.
  ivec2 px = ivec2(gl_FragCoord.xy);

  if (pc.mode == 1) {
    vec3 N = texelFetch(normalTxt, px, 0).xyz;
    fragColor = vec4(visualizeNormal(N), 1.0);
    return;
  }

  if (pc.mode == 2) {
    // Linear (Euclidean) depth wrote in closest-hit: visualize with a simple scale.
    float d = texelFetch(uDepth, px, 0).r;
    const float ZMAX = 100.0;       // matches your camera far; tweak if needed
    float v = 1.0 - clamp(d / ZMAX, 0.0, 1.0);
    fragColor = vec4(v, v, v, 1.0);
    return;
  }

  if (pc.mode == 3) {
    vec2 motion = texelFetch(uMotion, ivec2(gl_FragCoord.xy), 0).rg;

    // Hardcoded scale
    float motionScale = 32.0;  
    vec2 m = motion / motionScale;

    // Visualize X in red, Y in green
    fragColor = vec4(m, 0.0, 1.0);
    return;
   }

  if (pc.mode == 4) {
    float vis = texture(uNoisyShadow, outUV).r;  // 0..1 visibility/occlusion
    fragColor = vec4(vis, vis, vis, 1.0);       // grayscale
    return;
  } 
  
  if (pc.mode == 5) {
    float vis = texture(uDenoisedShadow, outUV).r;
    fragColor = vec4(vis, vis, vis, 1.0);
    return;
  }

  // Default: show color with gamma (linear -> display)
  vec4 col = texture(noisyTxt, outUV);
  fragColor = pow(col, vec4(1.0/2.2));
}

