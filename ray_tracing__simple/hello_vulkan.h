/*
 * Copyright (c) 2014-2021, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "nvvkhl/appbase_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/memallocator_dma_vk.hpp"
#include "nvvk/resourceallocator_vk.hpp"
#include "shaders/host_device.h"

// #VKRay
#include "nvvk/raytraceKHR_vk.hpp"

//--------------------------------------------------------------------------------------------------
// Simple rasterizer of OBJ objects
// - Each OBJ loaded are stored in an `ObjModel` and referenced by a `ObjInstance`
// - It is possible to have many `ObjInstance` referencing the same `ObjModel`
// - Rendering is done in an offscreen framebuffer
// - The image of the framebuffer is displayed in post-process in a full-screen quad
//
class HelloVulkan : public nvvkhl::AppBaseVk
{
public:
  void setup(const VkInstance& instance, const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t queueFamily) override;
  void createDescriptorSetLayout();
  void createGraphicsPipeline();
  void loadModel(const std::string& filename, glm::mat4 transform = glm::mat4(1));
  void updateDescriptorSet();
  void createUniformBuffer();
  void createObjDescriptionBuffer();
  void createTextureImages(const VkCommandBuffer& cmdBuf, const std::vector<std::string>& textures);
  void updateUniformBuffer(const VkCommandBuffer& cmdBuf);
  void onResize(int /*w*/, int /*h*/) override;
  void destroyResources();
  void rasterize(const VkCommandBuffer& cmdBuff);
  void onKeyboard(int key, int /*scancode*/, int action, int /*mods*/);

  // The OBJ model
  struct ObjModel
  {
    uint32_t     nbIndices{0};
    uint32_t     nbVertices{0};
    nvvk::Buffer vertexBuffer;    // Device buffer of all 'Vertex'
    nvvk::Buffer indexBuffer;     // Device buffer of the indices forming triangles
    nvvk::Buffer matColorBuffer;  // Device buffer of array of 'Wavefront material'
    nvvk::Buffer matIndexBuffer;  // Device buffer of array of 'Wavefront material'
  };

  struct ObjInstance
  {
    glm::mat4 transform;    // Matrix of the instance
    uint32_t  objIndex{0};  // Model index reference
  };

  // push constant for post draw
  struct PostPC 
  { 
    float aspect; 
    int mode; // 0 color, 1 normal, 2 depth
  };

  PostPC m_postPC{};

  // Information pushed at each draw call
  PushConstantRaster m_pcRaster{
      {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1},  // Identity matrix
      {10.f, 15.f, 8.f},                                 // light position
      0,                                                 // instance Id
      100.f,                                             // light intensity
      2,                                                  // light type
	  { 5.f, 0, 0, 0 }, // u vector
      { 0, 0, 5.f, 0 }, // v vector
	  25 // area
  };

  // Array of objects and instances in the scene
  std::vector<ObjModel>    m_objModel;   // Model on host
  std::vector<ObjDesc>     m_objDesc;    // Model description for device access
  std::vector<ObjInstance> m_instances;  // Scene model instances


  // Graphic pipeline
  VkPipelineLayout            m_pipelineLayout;
  VkPipeline                  m_graphicsPipeline;
  nvvk::DescriptorSetBindings m_descSetLayoutBind;
  VkDescriptorPool            m_descPool;
  VkDescriptorSetLayout       m_descSetLayout;
  VkDescriptorSet             m_descSet;

  nvvk::Buffer m_bGlobals;  // Device-Host of the camera matrices
  nvvk::Buffer m_bObjDesc;  // Device buffer of the OBJ descriptions

  std::vector<nvvk::Texture> m_textures;  // vector of all textures of the scene


  nvvk::ResourceAllocatorDma m_alloc;  // Allocator for buffer, images, acceleration structures
  nvvk::DebugUtil            m_debug;  // Utility to name objects


  // #Post - Draw the rendered image on a quad using a tonemapper
  void createOffscreenRender();
  void createPostPipeline();
  void createPostDescriptor();
  void updatePostDescriptorSet();
  void drawPost(VkCommandBuffer cmdBuf);

  nvvk::DescriptorSetBindings m_postDescSetLayoutBind;
  VkDescriptorPool            m_postDescPool{VK_NULL_HANDLE};
  VkDescriptorSetLayout       m_postDescSetLayout{VK_NULL_HANDLE};
  VkDescriptorSet             m_postDescSet{VK_NULL_HANDLE};
  VkPipeline                  m_postPipeline{VK_NULL_HANDLE};
  VkPipelineLayout            m_postPipelineLayout{VK_NULL_HANDLE};
  VkRenderPass                m_offscreenRenderPass{VK_NULL_HANDLE};
  VkFramebuffer               m_offscreenFramebuffer{VK_NULL_HANDLE};
  nvvk::Texture               m_offscreenColor{};
  nvvk::Texture               m_offscreenDepth{};
  nvvk::Texture               m_offscreenNormal{};
  nvvk::Texture               m_offscreenLinearDepth{};
  nvvk::Texture               m_offscreenMotion{};
  nvvk::Texture               m_offscreenNoisyShadow;   // R16F noisy visibility (0..1)

  VkFormat                    m_offscreenNoisyShadowFormat{ VK_FORMAT_R16_SFLOAT };
  VkFormat                    m_offscreenColorFormat{VK_FORMAT_R32G32B32A32_SFLOAT};
  VkFormat                    m_offscreenDepthFormat{VK_FORMAT_X8_D24_UNORM_PACK32};
  VkFormat                    m_offscreenNormalFormat{VK_FORMAT_R16G16B16A16_SFLOAT};
  VkFormat                    m_offscreenLinearDepthFormat{ VK_FORMAT_R32_SFLOAT };
  VkFormat                    m_offscreenMotionFormat{ VK_FORMAT_R16G16_SFLOAT };

  // Temporal Pass 1
  nvvk::Texture m_histMoments[2];           // ping-pong (R=mean, G=m2)
  int           m_histIdx = 0;              // which one is “current write”
  nvvk::Texture m_prevNormal;               // prev frame normal
  nvvk::Texture m_prevDepth;                // prev frame linear depth

  std::vector<VkDescriptorSet> m_temporalDescSets;
  VkPipelineLayout              m_temporalPipeLayout{};
  VkPipeline                    m_temporalPipeline{};
  nvvk::DescriptorSetBindings   m_temporalDSBind;
  VkDescriptorSetLayout         m_temporalDescSetLayout{};
  VkDescriptorPool              m_temporalDescPool{};
  VkDescriptorSet               m_temporalDescSet{};

  // how many frames you pipeline; usually your swapchain image count
  uint32_t m_framesInFlight = 2;  // set at init from swapchain

  void createTemporalDescriptor();
  void createTemporalPipeline();
  void updateTemporalDescriptorSet(int readIdx, int writeIdx, uint32_t frameIdx);
  void runTemporalPass1(VkCommandBuffer cmd, uint32_t frameIdx);

  // Pass 2
  // Spatial pass 2 outputs
  nvvk::Texture m_denoisedShadow;

  // Spatial (variance-guided bilateral filter)
  nvvk::DescriptorSetBindings m_spatialDSBind;
  VkDescriptorSetLayout       m_spatialDescSetLayout{ VK_NULL_HANDLE };
  VkDescriptorPool            m_spatialDescPool{ VK_NULL_HANDLE };
  std::vector<VkDescriptorSet> m_spatialDescSets;   // per-frame DS
  VkPipelineLayout            m_spatialPipeLayout{ VK_NULL_HANDLE };
  VkPipeline                  m_spatialPipeline{ VK_NULL_HANDLE };

  // Pass 2 (spatial bilateral filter)
  void createSpatialDescriptor();
  void createSpatialPipeline();
  void updateSpatialDescriptorSet(int readIdx, uint32_t frameIdx);
  void runSpatialPass2(VkCommandBuffer cmd, int readIdx, uint32_t frameIdx);

  // #VKRay
  void initRayTracing();
  auto objectToVkGeometryKHR(const ObjModel& model);
  void createBottomLevelAS();
  void createTopLevelAS();
  void createRtDescriptorSet();
  void updateRtDescriptorSet();
  void createRtPipeline();
  void createRtShaderBindingTable();
  void raytrace(const VkCommandBuffer& cmdBuf, const glm::vec4& clearColor);


  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  nvvk::RaytracingBuilderKHR                        m_rtBuilder;
  nvvk::DescriptorSetBindings                       m_rtDescSetLayoutBind;
  VkDescriptorPool                                  m_rtDescPool;
  VkDescriptorSetLayout                             m_rtDescSetLayout;
  VkDescriptorSet                                   m_rtDescSet;
  std::vector<VkRayTracingShaderGroupCreateInfoKHR> m_rtShaderGroups;
  VkPipelineLayout                                  m_rtPipelineLayout;
  VkPipeline                                        m_rtPipeline;

  nvvk::Buffer                    m_rtSBTBuffer;
  VkStridedDeviceAddressRegionKHR m_rgenRegion{};
  VkStridedDeviceAddressRegionKHR m_missRegion{};
  VkStridedDeviceAddressRegionKHR m_hitRegion{};
  VkStridedDeviceAddressRegionKHR m_callRegion{};

  // Push constant for ray tracer
  PushConstantRay m_pcRay{};

  // Camera reprojection state (for motion vectors / temporal denoise)
  mat4 m_prevV{ 1.0f };
  mat4 m_prevP{ 1.0f };
  mat4 m_prevVP{ 1.0f };
  mat4 m_currV{ 1.0f };
  mat4 m_currP{ 1.0f };
  mat4 m_currVP{ 1.0f };
  bool m_firstFrame = true;

  // Denoiser params
  struct SpatialParams {
	  float invSigmaZ = 60.0f;
	  float invSigmaN = 32.0f;
	  float varToRadius = 8.0f;
	  float varClamp = 0.25f;
  };

  SpatialParams m_spatialPC;

};
