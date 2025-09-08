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


#include <sstream>


#define STB_IMAGE_IMPLEMENTATION
#include "obj_loader.h"
#include "stb_image.h"

#include "hello_vulkan.h"
#include "nvh/alignment.hpp"
#include "nvh/cameramanipulator.hpp"
#include "nvh/fileoperations.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#include "nvvk/buffers_vk.hpp"

extern std::vector<std::string> defaultSearchPaths;

//--------------------------------------------------------------------------------------------------
// Keep the handle on the device
// Initialize the tool to do all our allocations: buffers, images
//
void HelloVulkan::setup(const VkInstance& instance, const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t queueFamily)
{
  AppBaseVk::setup(instance, device, physicalDevice, queueFamily);
  m_alloc.init(instance, device, physicalDevice);
  m_debug.setup(m_device);
  m_offscreenDepthFormat = nvvk::findDepthFormat(physicalDevice);
}

//--------------------------------------------------------------------------------------------------
// Called at each frame to update the camera matrix
//
void HelloVulkan::updateUniformBuffer(const VkCommandBuffer& cmdBuf)
{
  CameraManip.updateAnim();
  CameraManip.setAnimationDuration(0.0);  // makes it instant
  CameraManip.setSpeed(100.0f);            // try 50 or higher

  // Prepare new UBO contents on host.
  const float    aspectRatio = m_size.width / static_cast<float>(m_size.height);
  GlobalUniforms hostUBO     = {};
  const auto&    view        = CameraManip.getMatrix();
  glm::mat4      proj        = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), aspectRatio, 0.1f, 1000.0f);
  proj[1][1] *= -1;  // Inverting Y for Vulkan (not needed with perspectiveVK).

  if (m_firstFrame) {
	  m_prevV = view;
	  m_prevP = proj;
	  m_prevVP = proj * view;
	  m_firstFrame = false;
  }
  else {
	  m_prevV = m_currV;
	  m_prevP = m_currP;
	  m_prevVP = m_currVP;
  }
  m_currV = view;
  m_currP = proj;
  m_currVP = proj * view;

  hostUBO.VP_prev = m_prevVP;
  hostUBO.V_curr = m_currV;
  hostUBO.P_curr = m_currP;
  hostUBO.viewProj    = proj * view;
  hostUBO.viewInverse = glm::inverse(view);
  hostUBO.projInverse = glm::inverse(proj);
  hostUBO.viewportSize = glm::vec2(float(m_size.width), float(m_size.height));

  // UBO on the device, and what stages access it.
  VkBuffer deviceUBO      = m_bGlobals.buffer;
  auto     uboUsageStages = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;

  // Ensure that the modified UBO is not visible to previous frames.
  VkBufferMemoryBarrier beforeBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  beforeBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
  beforeBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  beforeBarrier.buffer        = deviceUBO;
  beforeBarrier.offset        = 0;
  beforeBarrier.size          = sizeof(hostUBO);
  vkCmdPipelineBarrier(cmdBuf, uboUsageStages, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_DEPENDENCY_DEVICE_GROUP_BIT, 0,
                       nullptr, 1, &beforeBarrier, 0, nullptr);


  // Schedule the host-to-device upload. (hostUBO is copied into the cmd
  // buffer so it is okay to deallocate when the function returns).
  vkCmdUpdateBuffer(cmdBuf, m_bGlobals.buffer, 0, sizeof(GlobalUniforms), &hostUBO);

  // Making sure the updated UBO will be visible.
  VkBufferMemoryBarrier afterBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  afterBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  afterBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  afterBarrier.buffer        = deviceUBO;
  afterBarrier.offset        = 0;
  afterBarrier.size          = sizeof(hostUBO);
  vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT, uboUsageStages, VK_DEPENDENCY_DEVICE_GROUP_BIT, 0,
                       nullptr, 1, &afterBarrier, 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Describing the layout pushed when rendering
//
void HelloVulkan::createDescriptorSetLayout()
{
  auto nbTxt = static_cast<uint32_t>(m_textures.size());

  // Camera matrices
  m_descSetLayoutBind.addBinding(SceneBindings::eGlobals, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                                 VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
  // Obj descriptions
  m_descSetLayoutBind.addBinding(SceneBindings::eObjDescs, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                 VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
  // Textures
  m_descSetLayoutBind.addBinding(SceneBindings::eTextures, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, nbTxt,
                                 VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);


  m_descSetLayout = m_descSetLayoutBind.createLayout(m_device);
  m_descPool      = m_descSetLayoutBind.createPool(m_device, 1);
  m_descSet       = nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout);
}

//--------------------------------------------------------------------------------------------------
// Setting up the buffers in the descriptor set
//
void HelloVulkan::updateDescriptorSet()
{
  std::vector<VkWriteDescriptorSet> writes;

  // Camera matrices and scene description
  VkDescriptorBufferInfo dbiUnif{m_bGlobals.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, SceneBindings::eGlobals, &dbiUnif));

  VkDescriptorBufferInfo dbiSceneDesc{m_bObjDesc.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, SceneBindings::eObjDescs, &dbiSceneDesc));

  // All texture samplers
  std::vector<VkDescriptorImageInfo> diit;
  for(auto& texture : m_textures)
  {
    diit.emplace_back(texture.descriptor);
  }
  writes.emplace_back(m_descSetLayoutBind.makeWriteArray(m_descSet, SceneBindings::eTextures, diit.data()));

  // Writing the information
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Creating the pipeline layout
//
void HelloVulkan::createGraphicsPipeline()
{
  VkPushConstantRange pushConstantRanges = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstantRaster)};

  // Creating the Pipeline Layout
  VkPipelineLayoutCreateInfo createInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  createInfo.setLayoutCount         = 1;
  createInfo.pSetLayouts            = &m_descSetLayout;
  createInfo.pushConstantRangeCount = 1;
  createInfo.pPushConstantRanges    = &pushConstantRanges;
  vkCreatePipelineLayout(m_device, &createInfo, nullptr, &m_pipelineLayout);


  // Creating the Pipeline
  std::vector<std::string>                paths = defaultSearchPaths;
  nvvk::GraphicsPipelineGeneratorCombined gpb(m_device, m_pipelineLayout, m_offscreenRenderPass);
  gpb.depthStencilState.depthTestEnable = true;
  gpb.addShader(nvh::loadFile("spv/vert_shader.vert.spv", true, paths, true), VK_SHADER_STAGE_VERTEX_BIT);
  gpb.addShader(nvh::loadFile("spv/frag_shader.frag.spv", true, paths, true), VK_SHADER_STAGE_FRAGMENT_BIT);
  gpb.addBindingDescription({0, sizeof(VertexObj)});
  gpb.addAttributeDescriptions({
      {0, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(VertexObj, pos))},
      {1, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(VertexObj, nrm))},
      {2, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(VertexObj, color))},
      {3, 0, VK_FORMAT_R32G32_SFLOAT, static_cast<uint32_t>(offsetof(VertexObj, texCoord))},
  });

  m_graphicsPipeline = gpb.createPipeline();
  m_debug.setObjectName(m_graphicsPipeline, "Graphics");
}

//--------------------------------------------------------------------------------------------------
// Loading the OBJ file and setting up all buffers
//
void HelloVulkan::loadModel(const std::string& filename, glm::mat4 transform)
{
  LOGI("Loading File:  %s \n", filename.c_str());
  ObjLoader loader;
  loader.loadModel(filename);

  // Converting from Srgb to linear
  for(auto& m : loader.m_materials)
  {
    m.ambient  = glm::pow(m.ambient, glm::vec3(2.2f));
    m.diffuse  = glm::pow(m.diffuse, glm::vec3(2.2f));
    m.specular = glm::pow(m.specular, glm::vec3(2.2f));
  }

  ObjModel model;
  model.nbIndices  = static_cast<uint32_t>(loader.m_indices.size());
  model.nbVertices = static_cast<uint32_t>(loader.m_vertices.size());

  // Create the buffers on Device and copy vertices, indices and materials
  nvvk::CommandPool  cmdBufGet(m_device, m_graphicsQueueIndex);
  VkCommandBuffer    cmdBuf          = cmdBufGet.createCommandBuffer();
  VkBufferUsageFlags flag            = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  VkBufferUsageFlags rayTracingFlags =  // used also for building acceleration structures
      flag | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  model.vertexBuffer = m_alloc.createBuffer(cmdBuf, loader.m_vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | rayTracingFlags);
  model.indexBuffer = m_alloc.createBuffer(cmdBuf, loader.m_indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | rayTracingFlags);
  model.matColorBuffer = m_alloc.createBuffer(cmdBuf, loader.m_materials, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | flag);
  model.matIndexBuffer = m_alloc.createBuffer(cmdBuf, loader.m_matIndx, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | flag);
  // Creates all textures found and find the offset for this model
  auto txtOffset = static_cast<uint32_t>(m_textures.size());
  createTextureImages(cmdBuf, loader.m_textures);
  cmdBufGet.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();

  std::string objNb = std::to_string(m_objModel.size());
  m_debug.setObjectName(model.vertexBuffer.buffer, (std::string("vertex_" + objNb)));
  m_debug.setObjectName(model.indexBuffer.buffer, (std::string("index_" + objNb)));
  m_debug.setObjectName(model.matColorBuffer.buffer, (std::string("mat_" + objNb)));
  m_debug.setObjectName(model.matIndexBuffer.buffer, (std::string("matIdx_" + objNb)));

  // Keeping transformation matrix of the instance
  ObjInstance instance;
  instance.transform = transform;
  instance.objIndex  = static_cast<uint32_t>(m_objModel.size());
  m_instances.push_back(instance);

  // Creating information for device access
  ObjDesc desc;
  desc.txtOffset            = txtOffset;
  desc.vertexAddress        = nvvk::getBufferDeviceAddress(m_device, model.vertexBuffer.buffer);
  desc.indexAddress         = nvvk::getBufferDeviceAddress(m_device, model.indexBuffer.buffer);
  desc.materialAddress      = nvvk::getBufferDeviceAddress(m_device, model.matColorBuffer.buffer);
  desc.materialIndexAddress = nvvk::getBufferDeviceAddress(m_device, model.matIndexBuffer.buffer);

  // Keeping the obj host model and device description
  m_objModel.emplace_back(model);
  m_objDesc.emplace_back(desc);
}


//--------------------------------------------------------------------------------------------------
// Creating the uniform buffer holding the camera matrices
// - Buffer is host visible
//
void HelloVulkan::createUniformBuffer()
{
  m_bGlobals = m_alloc.createBuffer(sizeof(GlobalUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_debug.setObjectName(m_bGlobals.buffer, "Globals");
}

//--------------------------------------------------------------------------------------------------
// Create a storage buffer containing the description of the scene elements
// - Which geometry is used by which instance
// - Transformation
// - Offset for texture
//
void HelloVulkan::createObjDescriptionBuffer()
{
  nvvk::CommandPool cmdGen(m_device, m_graphicsQueueIndex);

  auto cmdBuf = cmdGen.createCommandBuffer();
  m_bObjDesc  = m_alloc.createBuffer(cmdBuf, m_objDesc, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  cmdGen.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();
  m_debug.setObjectName(m_bObjDesc.buffer, "ObjDescs");
}

//--------------------------------------------------------------------------------------------------
// Creating all textures and samplers
//
void HelloVulkan::createTextureImages(const VkCommandBuffer& cmdBuf, const std::vector<std::string>& textures)
{
  VkSamplerCreateInfo samplerCreateInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  samplerCreateInfo.minFilter  = VK_FILTER_LINEAR;
  samplerCreateInfo.magFilter  = VK_FILTER_LINEAR;
  samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerCreateInfo.maxLod     = FLT_MAX;

  VkFormat format = VK_FORMAT_R8G8B8A8_SRGB;

  // If no textures are present, create a dummy one to accommodate the pipeline layout
  if(textures.empty() && m_textures.empty())
  {
    nvvk::Texture texture;

    std::array<uint8_t, 4> color{255u, 255u, 255u, 255u};
    VkDeviceSize           bufferSize      = sizeof(color);
    auto                   imgSize         = VkExtent2D{1, 1};
    auto                   imageCreateInfo = nvvk::makeImage2DCreateInfo(imgSize, format);

    // Creating the dummy texture
    nvvk::Image           image  = m_alloc.createImage(cmdBuf, bufferSize, color.data(), imageCreateInfo);
    VkImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
    texture                      = m_alloc.createTexture(image, ivInfo, samplerCreateInfo);

    // The image format must be in VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    nvvk::cmdBarrierImageLayout(cmdBuf, texture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_textures.push_back(texture);
  }
  else
  {
    // Uploading all images
    for(const auto& texture : textures)
    {
      std::stringstream o;
      int               texWidth, texHeight, texChannels;
      o << "media/textures/" << texture;
      std::string txtFile = nvh::findFile(o.str(), defaultSearchPaths, true);

      stbi_uc* stbi_pixels = stbi_load(txtFile.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

      std::array<stbi_uc, 4> color{255u, 0u, 255u, 255u};

      stbi_uc* pixels = stbi_pixels;
      // Handle failure
      if(!stbi_pixels)
      {
        texWidth = texHeight = 1;
        texChannels          = 4;
        pixels               = reinterpret_cast<stbi_uc*>(color.data());
      }

      VkDeviceSize bufferSize      = static_cast<uint64_t>(texWidth) * texHeight * sizeof(uint8_t) * 4;
      auto         imgSize         = VkExtent2D{(uint32_t)texWidth, (uint32_t)texHeight};
      auto         imageCreateInfo = nvvk::makeImage2DCreateInfo(imgSize, format, VK_IMAGE_USAGE_SAMPLED_BIT, true);

      {
        nvvk::Image image = m_alloc.createImage(cmdBuf, bufferSize, pixels, imageCreateInfo);
        nvvk::cmdGenerateMipmaps(cmdBuf, image.image, format, imgSize, imageCreateInfo.mipLevels);
        VkImageViewCreateInfo ivInfo  = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
        nvvk::Texture         texture = m_alloc.createTexture(image, ivInfo, samplerCreateInfo);

        m_textures.push_back(texture);
      }

      stbi_image_free(stbi_pixels);
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Destroying all allocations
//
void HelloVulkan::destroyResources()
{
  vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
  vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_descSetLayout, nullptr);

  vkDestroyPipeline(m_device, m_temporalPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_temporalPipeLayout, nullptr);
  vkDestroyDescriptorPool(m_device, m_temporalDescPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_temporalDescSetLayout, nullptr);

  vkDestroyPipeline(m_device, m_spatialPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_spatialPipeLayout, nullptr);
  vkDestroyDescriptorPool(m_device, m_spatialDescPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_spatialDescSetLayout, nullptr);

  m_alloc.destroy(m_bGlobals);
  m_alloc.destroy(m_bObjDesc);

  for(auto& m : m_objModel)
  {
    m_alloc.destroy(m.vertexBuffer);
    m_alloc.destroy(m.indexBuffer);
    m_alloc.destroy(m.matColorBuffer);
    m_alloc.destroy(m.matIndexBuffer);
  }

  for(auto& t : m_textures)
  {
    m_alloc.destroy(t);
  }

  //#Post
  m_alloc.destroy(m_offscreenNoisyShadow);
  m_alloc.destroy(m_offscreenColor);
  m_alloc.destroy(m_offscreenNormal);
  m_alloc.destroy(m_offscreenDepth);
  m_alloc.destroy(m_offscreenLinearDepth);
  m_alloc.destroy(m_offscreenMotion);
  m_alloc.destroy(m_prevNormal);
  m_alloc.destroy(m_prevDepth);
  m_alloc.destroy(m_denoisedShadow);

  // if you created history images
  for (int i = 0; i < 2; i++)
	  m_alloc.destroy(m_histMoments[i]);
  vkDestroyPipeline(m_device, m_postPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_postPipelineLayout, nullptr);
  vkDestroyDescriptorPool(m_device, m_postDescPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_postDescSetLayout, nullptr);
  vkDestroyRenderPass(m_device, m_offscreenRenderPass, nullptr);
  vkDestroyFramebuffer(m_device, m_offscreenFramebuffer, nullptr);


  // #VKRay
  m_rtBuilder.destroy();
  vkDestroyPipeline(m_device, m_rtPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_rtPipelineLayout, nullptr);
  vkDestroyDescriptorPool(m_device, m_rtDescPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_rtDescSetLayout, nullptr);
  m_alloc.destroy(m_rtSBTBuffer);

  m_alloc.deinit();
}

//--------------------------------------------------------------------------------------------------
// Drawing the scene in raster mode
//
void HelloVulkan::rasterize(const VkCommandBuffer& cmdBuf)
{
  VkDeviceSize offset{0};

  m_debug.beginLabel(cmdBuf, "Rasterize");

  // Dynamic Viewport
  setViewport(cmdBuf);

  // Drawing all triangles
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descSet, 0, nullptr);


  for(const HelloVulkan::ObjInstance& inst : m_instances)
  {
    auto& model            = m_objModel[inst.objIndex];
    m_pcRaster.objIndex    = inst.objIndex;  // Telling which object is drawn
    m_pcRaster.modelMatrix = inst.transform;

    vkCmdPushConstants(cmdBuf, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(PushConstantRaster), &m_pcRaster);
    vkCmdBindVertexBuffers(cmdBuf, 0, 1, &model.vertexBuffer.buffer, &offset);
    vkCmdBindIndexBuffer(cmdBuf, model.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmdBuf, model.nbIndices, 1, 0, 0, 0);
  }
  m_debug.endLabel(cmdBuf);
}

//--------------------------------------------------------------------------------------------------
// Handling resize of the window
//
void HelloVulkan::onResize(int /*w*/, int /*h*/)
{
  createOffscreenRender();
  updatePostDescriptorSet();
  updateRtDescriptorSet();
  m_firstFrame = true;  // <-- ensure VP_prev == VP_curr next frame (no bogus motion)
}

//////////////////////////////////////////////////////////////////////////
// Post-processing
//////////////////////////////////////////////////////////////////////////


//--------------------------------------------------------------------------------------------------
// Creating an offscreen frame buffer and the associated render pass
//
void HelloVulkan::createOffscreenRender()
{
	m_alloc.destroy(m_offscreenColor);
	m_alloc.destroy(m_offscreenDepth);
    m_alloc.destroy(m_offscreenNormal);
    m_alloc.destroy(m_offscreenLinearDepth);
    m_alloc.destroy(m_offscreenMotion);
	m_alloc.destroy(m_prevNormal);
	m_alloc.destroy(m_prevDepth);

	// if you created history images
	for (int i = 0; i < 2; i++)
		m_alloc.destroy(m_histMoments[i]);

  // Creating the color image
  {
    auto colorCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_offscreenColorFormat,
                                                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                                                           | VK_IMAGE_USAGE_STORAGE_BIT);


    nvvk::Image           image  = m_alloc.createImage(colorCreateInfo);
    VkImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, colorCreateInfo);
    VkSamplerCreateInfo   sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    m_offscreenColor                        = m_alloc.createTexture(image, ivInfo, sampler);
    m_offscreenColor.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }

  // Noisy shadow (scalar visibility) image
  {
	  auto ci = nvvk::makeImage2DCreateInfo(
		  m_size, m_offscreenNoisyShadowFormat,
		  VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

	  nvvk::Image img = m_alloc.createImage(ci);
	  VkImageViewCreateInfo iv = nvvk::makeImageViewCreateInfo(img.image, ci);
	  VkSamplerCreateInfo   sp{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
	  m_offscreenNoisyShadow = m_alloc.createTexture(img, iv, sp);
	  m_offscreenNoisyShadow.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }

  // Create normal image
  {
	  auto normalCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_offscreenNormalFormat,
		                                                  VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

	  nvvk::Image image = m_alloc.createImage(normalCreateInfo);
	  VkImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, normalCreateInfo);
	  VkSamplerCreateInfo sampler{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
	  m_offscreenNormal = m_alloc.createTexture(image, ivInfo, sampler);
	  m_offscreenNormal.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }

  // Creating the depth buffer
  auto depthCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_offscreenDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
  {
    nvvk::Image image = m_alloc.createImage(depthCreateInfo);


    VkImageViewCreateInfo depthStencilView{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    depthStencilView.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    depthStencilView.format           = m_offscreenDepthFormat;
    depthStencilView.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    depthStencilView.image            = image.image;

    m_offscreenDepth = m_alloc.createTexture(image, depthStencilView);
  }
  
  // Create the linear depth buffer for RT
  {
	  auto linearDepthCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_offscreenLinearDepthFormat,
		                                                       VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

	  nvvk::Image image = m_alloc.createImage(linearDepthCreateInfo);
	  VkImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, linearDepthCreateInfo);
	  VkSamplerCreateInfo sampler{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
	  m_offscreenLinearDepth = m_alloc.createTexture(image, ivInfo, sampler);
      m_offscreenLinearDepth.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }

  // Motion vector for RT
  {
	  auto motionCreateInfo = nvvk::makeImage2DCreateInfo(
		  m_size, m_offscreenMotionFormat,
		  VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

	  nvvk::Image image = m_alloc.createImage(motionCreateInfo);
	  VkImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, motionCreateInfo);
	  VkSamplerCreateInfo sampler{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
	  m_offscreenMotion = m_alloc.createTexture(image, ivInfo, sampler);
	  m_offscreenMotion.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }

  // Moment images
  {
      auto momentsCI = nvvk::makeImage2DCreateInfo(
	      m_size, VK_FORMAT_R16G16_SFLOAT,
	      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

      for (int i = 0; i < 2; i++) {
          nvvk::Image img = m_alloc.createImage(momentsCI);
          VkImageViewCreateInfo iv = nvvk::makeImageViewCreateInfo(img.image, momentsCI);
          VkSamplerCreateInfo   sp{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
          m_histMoments[i] = m_alloc.createTexture(img, iv, sp);
          m_histMoments[i].descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
      }
  }

   // Prev Normal (to store last frame's normal buffer)
   {
   	    auto ci = nvvk::makeImage2DCreateInfo(
   		    m_size,
   		    m_offscreenNormalFormat,  // e.g. VK_FORMAT_R16G16B16A16_SFLOAT
   		    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
   
   	    nvvk::Image img = m_alloc.createImage(ci);
   	    VkImageViewCreateInfo iv = nvvk::makeImageViewCreateInfo(img.image, ci);
   	    VkSamplerCreateInfo sp{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
   	    m_prevNormal = m_alloc.createTexture(img, iv, sp);
   	    m_prevNormal.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
   }
   
   // Prev Depth (to store last frame's linear depth buffer)
   {
   	    auto ci = nvvk::makeImage2DCreateInfo(
   		    m_size,
   		    m_offscreenLinearDepthFormat,  // e.g. VK_FORMAT_R32_SFLOAT
   		    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
   
   	    nvvk::Image img = m_alloc.createImage(ci);
   	    VkImageViewCreateInfo iv = nvvk::makeImageViewCreateInfo(img.image, ci);
   	    VkSamplerCreateInfo sp{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
   	    m_prevDepth = m_alloc.createTexture(img, iv, sp);
   	    m_prevDepth.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
   }

   // During createOffscreenRender()
   {
	   auto ci = nvvk::makeImage2DCreateInfo(m_size, VK_FORMAT_R16_SFLOAT,
		   VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
	   nvvk::Image img = m_alloc.createImage(ci);
	   VkImageViewCreateInfo iv = nvvk::makeImageViewCreateInfo(img.image, ci);
	   VkSamplerCreateInfo sp{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
	   m_denoisedShadow = m_alloc.createTexture(img, iv, sp);
	   m_denoisedShadow.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;


   }

  // Setting the image layout for both color and depth
  {
    nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
    auto              cmdBuf = genCmdBuf.createCommandBuffer();
	nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenNoisyShadow.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenColor.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
	nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenNormal.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenLinearDepth.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenDepth.image, VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
	nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenMotion.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

	for (int i = 0; i < 2; ++i) {
		nvvk::cmdBarrierImageLayout(cmdBuf, m_histMoments[i].image,
			VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
	}

	// init layout
	nvvk::cmdBarrierImageLayout(cmdBuf, m_denoisedShadow.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

	nvvk::cmdBarrierImageLayout(cmdBuf, m_prevNormal.image,
		VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
	nvvk::cmdBarrierImageLayout(cmdBuf, m_prevDepth.image,
		VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    genCmdBuf.submitAndWait(cmdBuf);
  }

  // Creating a renderpass for the offscreen
  if(!m_offscreenRenderPass)
  {
    m_offscreenRenderPass = nvvk::createRenderPass(m_device, {m_offscreenColorFormat}, m_offscreenDepthFormat, 1, true,
                                                   true, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
  }


  // Creating the frame buffer for offscreen
  std::vector<VkImageView> attachments = {m_offscreenColor.descriptor.imageView, m_offscreenDepth.descriptor.imageView};

  vkDestroyFramebuffer(m_device, m_offscreenFramebuffer, nullptr);
  VkFramebufferCreateInfo info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
  info.renderPass      = m_offscreenRenderPass;
  info.attachmentCount = 2;
  info.pAttachments    = attachments.data();
  info.width           = m_size.width;
  info.height          = m_size.height;
  info.layers          = 1;
  vkCreateFramebuffer(m_device, &info, nullptr, &m_offscreenFramebuffer);
}

//--------------------------------------------------------------------------------------------------
// The pipeline is how things are rendered, which shaders, type of primitives, depth test and more
//
void HelloVulkan::createPostPipeline()
{
  // Push constants in the fragment shader
  //VkPushConstantRange pushConstantRanges = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float)};
  VkPushConstantRange pcRange{ VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PostPC) };

  // Creating the pipeline layout
  VkPipelineLayoutCreateInfo createInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  createInfo.setLayoutCount         = 1;
  createInfo.pSetLayouts            = &m_postDescSetLayout;
  createInfo.pushConstantRangeCount = 1;
  createInfo.pPushConstantRanges    = &pcRange;
  vkCreatePipelineLayout(m_device, &createInfo, nullptr, &m_postPipelineLayout);


  // Pipeline: completely generic, no vertices
  nvvk::GraphicsPipelineGeneratorCombined pipelineGenerator(m_device, m_postPipelineLayout, m_renderPass);
  pipelineGenerator.addShader(nvh::loadFile("spv/passthrough.vert.spv", true, defaultSearchPaths, true), VK_SHADER_STAGE_VERTEX_BIT);
  pipelineGenerator.addShader(nvh::loadFile("spv/post.frag.spv", true, defaultSearchPaths, true), VK_SHADER_STAGE_FRAGMENT_BIT);
  pipelineGenerator.rasterizationState.cullMode = VK_CULL_MODE_NONE;
  m_postPipeline                                = pipelineGenerator.createPipeline();
  m_debug.setObjectName(m_postPipeline, "post");
}

//--------------------------------------------------------------------------------------------------
// The descriptor layout is the description of the data that is passed to the vertex or the
// fragment program.
//
void HelloVulkan::createPostDescriptor()
{
  m_postDescSetLayoutBind.addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT); // color
  m_postDescSetLayoutBind.addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT); // normal
  m_postDescSetLayoutBind.addBinding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT); // linear depth
  m_postDescSetLayoutBind.addBinding(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT); // motion
  m_postDescSetLayoutBind.addBinding(4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT); // noisy shadow  <-- NEW
  m_postDescSetLayoutBind.addBinding(5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT); // denoised     <-- NEW

  m_postDescSetLayout = m_postDescSetLayoutBind.createLayout(m_device);
  m_postDescPool      = m_postDescSetLayoutBind.createPool(m_device);
  m_postDescSet       = nvvk::allocateDescriptorSet(m_device, m_postDescPool, m_postDescSetLayout);
}


//--------------------------------------------------------------------------------------------------
// Update the output
//
void HelloVulkan::updatePostDescriptorSet()
{
  VkWriteDescriptorSet writes[6] = {
    m_postDescSetLayoutBind.makeWrite(m_postDescSet, 0, &m_offscreenColor.descriptor),
    m_postDescSetLayoutBind.makeWrite(m_postDescSet, 1, &m_offscreenNormal.descriptor),
    m_postDescSetLayoutBind.makeWrite(m_postDescSet, 2, &m_offscreenLinearDepth.descriptor),
    m_postDescSetLayoutBind.makeWrite(m_postDescSet, 3, &m_offscreenMotion.descriptor),
	m_postDescSetLayoutBind.makeWrite(m_postDescSet, 4, &m_offscreenNoisyShadow.descriptor), // NEW
    m_postDescSetLayoutBind.makeWrite(m_postDescSet, 5, &m_denoisedShadow.descriptor)        // NEW (optional)
  };
  vkUpdateDescriptorSets(m_device, 6, writes, 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Draw a full screen quad with the attached image
//
void HelloVulkan::drawPost(VkCommandBuffer cmdBuf)
{
  m_debug.beginLabel(cmdBuf, "Post");

  setViewport(cmdBuf);

  m_postPC.aspect = static_cast<float>(m_size.width) / static_cast<float>(m_size.height);
  //m_postPC.mode = 3; // 0 = color, 1 = normals, 2 = depth (or whatever you choose), 3 motion

  vkCmdPushConstants(cmdBuf, m_postPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
	  0, sizeof(PostPC), &m_postPC);

  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_postPipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_postPipelineLayout,
	  0, 1, &m_postDescSet, 0, nullptr);
  vkCmdDraw(cmdBuf, 3, 1, 0, 0);

  m_debug.endLabel(cmdBuf);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

//--------------------------------------------------------------------------------------------------
// Initialize Vulkan ray tracing
// #VKRay
void HelloVulkan::initRayTracing()
{
  // Requesting ray tracing properties
  VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  prop2.pNext = &m_rtProperties;
  vkGetPhysicalDeviceProperties2(m_physicalDevice, &prop2);

  m_rtBuilder.setup(m_device, &m_alloc, m_graphicsQueueIndex);
}

//--------------------------------------------------------------------------------------------------
// Convert an OBJ model into the ray tracing geometry used to build the BLAS
//
auto HelloVulkan::objectToVkGeometryKHR(const ObjModel& model)
{
  // BLAS builder requires raw device addresses.
  VkDeviceAddress vertexAddress = nvvk::getBufferDeviceAddress(m_device, model.vertexBuffer.buffer);
  VkDeviceAddress indexAddress  = nvvk::getBufferDeviceAddress(m_device, model.indexBuffer.buffer);

  uint32_t maxPrimitiveCount = model.nbIndices / 3;

  // Describe buffer as array of VertexObj.
  VkAccelerationStructureGeometryTrianglesDataKHR triangles{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
  triangles.vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT;  // vec3 vertex position data.
  triangles.vertexData.deviceAddress = vertexAddress;
  triangles.vertexStride             = sizeof(VertexObj);
  // Describe index data (32-bit unsigned int)
  triangles.indexType               = VK_INDEX_TYPE_UINT32;
  triangles.indexData.deviceAddress = indexAddress;
  // Indicate identity transform by setting transformData to null device pointer.
  //triangles.transformData = {};
  triangles.maxVertex = model.nbVertices - 1;

  // Identify the above data as containing opaque triangles.
  VkAccelerationStructureGeometryKHR asGeom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  asGeom.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  asGeom.flags              = VK_GEOMETRY_OPAQUE_BIT_KHR;
  asGeom.geometry.triangles = triangles;

  // The entire array will be used to build the BLAS.
  VkAccelerationStructureBuildRangeInfoKHR offset;
  offset.firstVertex     = 0;
  offset.primitiveCount  = maxPrimitiveCount;
  offset.primitiveOffset = 0;
  offset.transformOffset = 0;

  // Our blas is made from only one geometry, but could be made of many geometries
  nvvk::RaytracingBuilderKHR::BlasInput input;
  input.asGeometry.emplace_back(asGeom);
  input.asBuildOffsetInfo.emplace_back(offset);

  return input;
}

//--------------------------------------------------------------------------------------------------
//
//
void HelloVulkan::createBottomLevelAS()
{
  // BLAS - Storing each primitive in a geometry
  std::vector<nvvk::RaytracingBuilderKHR::BlasInput> allBlas;
  allBlas.reserve(m_objModel.size());
  for(const auto& obj : m_objModel)
  {
    auto blas = objectToVkGeometryKHR(obj);

    // We could add more geometry in each BLAS, but we add only one for now
    allBlas.emplace_back(blas);
  }
  m_rtBuilder.buildBlas(allBlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
}

//--------------------------------------------------------------------------------------------------
//
//
void HelloVulkan::createTopLevelAS()
{
  std::vector<VkAccelerationStructureInstanceKHR> tlas;
  tlas.reserve(m_instances.size());
  for(const HelloVulkan::ObjInstance& inst : m_instances)
  {
    VkAccelerationStructureInstanceKHR rayInst{};
    rayInst.transform                      = nvvk::toTransformMatrixKHR(inst.transform);  // Position of the instance
    rayInst.instanceCustomIndex            = inst.objIndex;                               // gl_InstanceCustomIndexEXT
    rayInst.accelerationStructureReference = m_rtBuilder.getBlasDeviceAddress(inst.objIndex);
    rayInst.flags                          = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    rayInst.mask                           = 0xFF;       //  Only be hit if rayMask & instance.mask != 0
    rayInst.instanceShaderBindingTableRecordOffset = 0;  // We will use the same hit group for all objects
    tlas.emplace_back(rayInst);
  }
  m_rtBuilder.buildTlas(tlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
}

//--------------------------------------------------------------------------------------------------
// This descriptor set holds the Acceleration structure and the output image
//
void HelloVulkan::createRtDescriptorSet()
{
  // Top-level acceleration structure, usable by both the ray generation and the closest hit (to shoot shadow rays)
  m_rtDescSetLayoutBind.addBinding(RtxBindings::eTlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1,
                                   VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);  // TLAS
  m_rtDescSetLayoutBind.addBinding(RtxBindings::eOutImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                                   VK_SHADER_STAGE_RAYGEN_BIT_KHR);  // Output image

  m_rtDescSetLayoutBind.addBinding(RtxBindings::eNormalImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
	                               VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR); // Output world normal

  m_rtDescSetLayoutBind.addBinding(RtxBindings::eLinearDepth, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
	                               VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR); // Output linear Depth

  m_rtDescSetLayoutBind.addBinding(RtxBindings::eMotionImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
	                               VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);

  m_rtDescSetLayoutBind.addBinding(RtxBindings::eNoisyShadow, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
								   VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR);

  m_rtDescPool      = m_rtDescSetLayoutBind.createPool(m_device);
  m_rtDescSetLayout = m_rtDescSetLayoutBind.createLayout(m_device);

  VkDescriptorSetAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  allocateInfo.descriptorPool     = m_rtDescPool;
  allocateInfo.descriptorSetCount = 1;
  allocateInfo.pSetLayouts        = &m_rtDescSetLayout;
  vkAllocateDescriptorSets(m_device, &allocateInfo, &m_rtDescSet);


  VkAccelerationStructureKHR tlas = m_rtBuilder.getAccelerationStructure();
  VkWriteDescriptorSetAccelerationStructureKHR descASInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
  descASInfo.accelerationStructureCount = 1;
  descASInfo.pAccelerationStructures    = &tlas;
  VkDescriptorImageInfo imageInfo{{}, m_offscreenColor.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL};
  VkDescriptorImageInfo normalInfo{{}, m_offscreenNormal.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL };
  VkDescriptorImageInfo linearDepthInfo{{}, m_offscreenLinearDepth.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL };
  VkDescriptorImageInfo motionInfo{ {}, m_offscreenMotion.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL };
  VkDescriptorImageInfo noisyInfo{ {}, m_offscreenNoisyShadow.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL };

  std::vector<VkWriteDescriptorSet> writes;
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eTlas, &descASInfo));
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eOutImage, &imageInfo));
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eNormalImage, &normalInfo));
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eLinearDepth, &linearDepthInfo));
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eMotionImage, &motionInfo));
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eNoisyShadow, &noisyInfo));

  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Writes the output image to the descriptor set
// - Required when changing resolution
//
void HelloVulkan::updateRtDescriptorSet()
{
  // (1) Output buffer
  VkDescriptorImageInfo imageInfo{{}, m_offscreenColor.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL};
  VkWriteDescriptorSet  wds = m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eOutImage, &imageInfo);
  vkUpdateDescriptorSets(m_device, 1, &wds, 0, nullptr);

  VkDescriptorImageInfo normalInfo{ {}, m_offscreenNormal.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL };
  auto w = m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eNormalImage, &normalInfo);
  vkUpdateDescriptorSets(m_device, 1, &w, 0, nullptr);

  VkDescriptorImageInfo linearDepthInfo{ {}, m_offscreenLinearDepth.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL };
  auto l = m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eLinearDepth, &linearDepthInfo);
  vkUpdateDescriptorSets(m_device, 1, &l, 0, nullptr);

  VkDescriptorImageInfo motionInfo{ {}, m_offscreenMotion.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL };
  auto m = m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eMotionImage, &motionInfo);
  vkUpdateDescriptorSets(m_device, 1, &m, 0, nullptr);

  VkDescriptorImageInfo noisyInfo{ {}, m_offscreenNoisyShadow.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL };
  auto ns = m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eNoisyShadow, &noisyInfo);
  vkUpdateDescriptorSets(m_device, 1, &ns, 0, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Pipeline for the ray tracer: all shaders, raygen, chit, miss
//
void HelloVulkan::createRtPipeline()
{
  enum StageIndices
  {
    eRaygen,
    eMiss,
    eMiss2,
    eClosestHit,
    eShaderGroupCount
  };

  // All stages
  std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
  VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stage.pName = "main";  // All the same entry point
  // Raygen
  stage.module = nvvk::createShaderModule(m_device, nvh::loadFile("spv/raytrace.rgen.spv", true, defaultSearchPaths, true));
  stage.stage     = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
  stages[eRaygen] = stage;
  // Miss
  stage.module = nvvk::createShaderModule(m_device, nvh::loadFile("spv/raytrace.rmiss.spv", true, defaultSearchPaths, true));
  stage.stage   = VK_SHADER_STAGE_MISS_BIT_KHR;
  stages[eMiss] = stage;
  // The second miss shader is invoked when a shadow ray misses the geometry. It simply indicates that no occlusion has been found
  stage.module =
      nvvk::createShaderModule(m_device, nvh::loadFile("spv/raytraceShadow.rmiss.spv", true, defaultSearchPaths, true));
  stage.stage    = VK_SHADER_STAGE_MISS_BIT_KHR;
  stages[eMiss2] = stage;
  // Hit Group - Closest Hit
  stage.module = nvvk::createShaderModule(m_device, nvh::loadFile("spv/raytrace.rchit.spv", true, defaultSearchPaths, true));
  stage.stage         = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
  stages[eClosestHit] = stage;


  // Shader groups
  VkRayTracingShaderGroupCreateInfoKHR group{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
  group.anyHitShader       = VK_SHADER_UNUSED_KHR;
  group.closestHitShader   = VK_SHADER_UNUSED_KHR;
  group.generalShader      = VK_SHADER_UNUSED_KHR;
  group.intersectionShader = VK_SHADER_UNUSED_KHR;

  // Raygen
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eRaygen;
  m_rtShaderGroups.push_back(group);

  // Miss
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMiss;
  m_rtShaderGroups.push_back(group);

  // Shadow Miss
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMiss2;
  m_rtShaderGroups.push_back(group);

  // closest hit shader
  group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
  group.generalShader    = VK_SHADER_UNUSED_KHR;
  group.closestHitShader = eClosestHit;
  m_rtShaderGroups.push_back(group);

  // Push constant: we want to be able to update constants used by the shaders
  VkPushConstantRange pushConstant{VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
                                   0, sizeof(PushConstantRay)};


  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
  pipelineLayoutCreateInfo.pPushConstantRanges    = &pushConstant;

  // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
  std::vector<VkDescriptorSetLayout> rtDescSetLayouts = {m_rtDescSetLayout, m_descSetLayout};
  pipelineLayoutCreateInfo.setLayoutCount             = static_cast<uint32_t>(rtDescSetLayouts.size());
  pipelineLayoutCreateInfo.pSetLayouts                = rtDescSetLayouts.data();

  vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &m_rtPipelineLayout);


  // Assemble the shader stages and recursion depth info into the ray tracing pipeline
  VkRayTracingPipelineCreateInfoKHR rayPipelineInfo{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
  rayPipelineInfo.stageCount = static_cast<uint32_t>(stages.size());  // Stages are shaders
  rayPipelineInfo.pStages    = stages.data();

  // In this case, m_rtShaderGroups.size() == 4: we have one raygen group,
  // two miss shader groups, and one hit group.
  rayPipelineInfo.groupCount = static_cast<uint32_t>(m_rtShaderGroups.size());
  rayPipelineInfo.pGroups    = m_rtShaderGroups.data();

  // The ray tracing process can shoot rays from the camera, and a shadow ray can be shot from the
  // hit points of the camera rays, hence a recursion level of 2. This number should be kept as low
  // as possible for performance reasons. Even recursive ray tracing should be flattened into a loop
  // in the ray generation to avoid deep recursion.
  rayPipelineInfo.maxPipelineRayRecursionDepth = 2;  // Ray depth
  rayPipelineInfo.layout                       = m_rtPipelineLayout;

  vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &rayPipelineInfo, nullptr, &m_rtPipeline);


  // Spec only guarantees 1 level of "recursion". Check for that sad possibility here.
  if(m_rtProperties.maxRayRecursionDepth <= 1)
  {
    throw std::runtime_error("Device fails to support ray recursion (m_rtProperties.maxRayRecursionDepth <= 1)");
  }

  for(auto& s : stages)
    vkDestroyShaderModule(m_device, s.module, nullptr);
}

//--------------------------------------------------------------------------------------------------
// The Shader Binding Table (SBT)
// - getting all shader handles and write them in a SBT buffer
// - Besides exception, this could be always done like this
//
void HelloVulkan::createRtShaderBindingTable()
{
  uint32_t missCount{2};
  uint32_t hitCount{1};
  auto     handleCount = 1 + missCount + hitCount;
  uint32_t handleSize  = m_rtProperties.shaderGroupHandleSize;

  // The SBT (buffer) need to have starting groups to be aligned and handles in the group to be aligned.
  uint32_t handleSizeAligned = nvh::align_up(handleSize, m_rtProperties.shaderGroupHandleAlignment);

  m_rgenRegion.stride = nvh::align_up(handleSizeAligned, m_rtProperties.shaderGroupBaseAlignment);
  m_rgenRegion.size = m_rgenRegion.stride;  // The size member of pRayGenShaderBindingTable must be equal to its stride member
  m_missRegion.stride = handleSizeAligned;
  m_missRegion.size   = nvh::align_up(missCount * handleSizeAligned, m_rtProperties.shaderGroupBaseAlignment);
  m_hitRegion.stride  = handleSizeAligned;
  m_hitRegion.size    = nvh::align_up(hitCount * handleSizeAligned, m_rtProperties.shaderGroupBaseAlignment);

  // Get the shader group handles
  uint32_t             dataSize = handleCount * handleSize;
  std::vector<uint8_t> handles(dataSize);
  auto result = vkGetRayTracingShaderGroupHandlesKHR(m_device, m_rtPipeline, 0, handleCount, dataSize, handles.data());
  assert(result == VK_SUCCESS);

  // Allocate a buffer for storing the SBT.
  VkDeviceSize sbtSize = m_rgenRegion.size + m_missRegion.size + m_hitRegion.size + m_callRegion.size;
  m_rtSBTBuffer        = m_alloc.createBuffer(sbtSize,
                                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                                  | VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR,
                                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  m_debug.setObjectName(m_rtSBTBuffer.buffer, std::string("SBT"));  // Give it a debug name for NSight.

  // Find the SBT addresses of each group
  VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, m_rtSBTBuffer.buffer};
  VkDeviceAddress           sbtAddress = vkGetBufferDeviceAddress(m_device, &info);
  m_rgenRegion.deviceAddress           = sbtAddress;
  m_missRegion.deviceAddress           = sbtAddress + m_rgenRegion.size;
  m_hitRegion.deviceAddress            = sbtAddress + m_rgenRegion.size + m_missRegion.size;

  // Helper to retrieve the handle data
  auto getHandle = [&](int i) { return handles.data() + i * handleSize; };

  // Map the SBT buffer and write in the handles.
  auto*    pSBTBuffer = reinterpret_cast<uint8_t*>(m_alloc.map(m_rtSBTBuffer));
  uint8_t* pData{nullptr};
  uint32_t handleIdx{0};
  // Raygen
  pData = pSBTBuffer;
  memcpy(pData, getHandle(handleIdx++), handleSize);
  // Miss
  pData = pSBTBuffer + m_rgenRegion.size;
  for(uint32_t c = 0; c < missCount; c++)
  {
    memcpy(pData, getHandle(handleIdx++), handleSize);
    pData += m_missRegion.stride;
  }
  // Hit
  pData = pSBTBuffer + m_rgenRegion.size + m_missRegion.size;
  for(uint32_t c = 0; c < hitCount; c++)
  {
    memcpy(pData, getHandle(handleIdx++), handleSize);
    pData += m_hitRegion.stride;
  }

  m_alloc.unmap(m_rtSBTBuffer);
  m_alloc.finalizeAndReleaseStaging();
}

static inline VkImageMemoryBarrier makeRtToFsBarrier(VkImage image) {
	VkImageMemoryBarrier ib{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
	ib.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;   // written by RT shaders
	ib.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;    // to be read by FS
	ib.oldLayout = VK_IMAGE_LAYOUT_GENERAL;      // we keep GENERAL
	ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	ib.image = image;
	ib.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
	return ib;
}

//--------------------------------------------------------------------------------------------------
// Ray Tracing the scene
//
void HelloVulkan::raytrace(const VkCommandBuffer& cmdBuf, const glm::vec4& clearColor)
{
  m_debug.beginLabel(cmdBuf, "Ray trace");
  // Initializing push constant values
  m_pcRay.clearColor     = clearColor;
  m_pcRay.lightPosition  = m_pcRaster.lightPosition;
  m_pcRay.lightIntensity = m_pcRaster.lightIntensity;
  m_pcRay.lightType      = m_pcRaster.lightType;
  // rectangle light
  m_pcRay.lightU = m_pcRaster.u;
  m_pcRay.lightV = m_pcRaster.v;
  m_pcRay.lightArea = m_pcRaster.area;

  std::vector<VkDescriptorSet> descSets{m_rtDescSet, m_descSet};
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipelineLayout, 0,
                          (uint32_t)descSets.size(), descSets.data(), 0, nullptr);
  vkCmdPushConstants(cmdBuf, m_rtPipelineLayout,
                     VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
                     0, sizeof(PushConstantRay), &m_pcRay);


  vkCmdTraceRaysKHR(cmdBuf, &m_rgenRegion, &m_missRegion, &m_hitRegion, &m_callRegion, m_size.width, m_size.height, 1);
  // Make RT writes visible to the fragment shader in the post pass
  std::vector<VkImageMemoryBarrier> barriers;
  barriers.push_back(makeRtToFsBarrier(m_offscreenColor.image));
  barriers.push_back(makeRtToFsBarrier(m_offscreenNormal.image));
  barriers.push_back(makeRtToFsBarrier(m_offscreenLinearDepth.image)); // if you created it
  barriers.push_back(makeRtToFsBarrier(m_offscreenMotion.image)); 
  barriers.push_back(makeRtToFsBarrier(m_offscreenNoisyShadow.image));

  if (!barriers.empty()) {
	  vkCmdPipelineBarrier(cmdBuf,
		  VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,   // src
		  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,          // dst
		  0,
		  0, nullptr, 0, nullptr,
		  static_cast<uint32_t>(barriers.size()), barriers.data());
  }

  m_debug.endLabel(cmdBuf);
}

void HelloVulkan::createTemporalDescriptor()
{
	m_temporalDSBind = {};
	// sampled inputs
	m_temporalDSBind.addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT); // histPrev
	m_temporalDSBind.addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT); // normalPrev
	m_temporalDSBind.addBinding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT); // depthPrev
	m_temporalDSBind.addBinding(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT); // normalCurr
	m_temporalDSBind.addBinding(4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT); // depthCurr
	m_temporalDSBind.addBinding(5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT); // motion
	m_temporalDSBind.addBinding(6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT); // noisyShadow
	// storage output
	m_temporalDSBind.addBinding(7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT); // outMoments

	m_temporalDescSetLayout = m_temporalDSBind.createLayout(m_device);
	m_temporalDescPool = m_temporalDSBind.createPool(m_device, m_framesInFlight);

	m_temporalDescSets.resize(m_framesInFlight);
	for (uint32_t i = 0; i < m_framesInFlight; ++i) {
		m_temporalDescSets[i] = nvvk::allocateDescriptorSet(m_device, m_temporalDescPool, m_temporalDescSetLayout);
	}
}

void HelloVulkan::createTemporalPipeline()
{
	// Push constants
	struct PC { float tauZ, tauN, clampK, alphaUse; int firstFrame; };
	VkPushConstantRange pcRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC) };

	VkPipelineLayoutCreateInfo pli{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
	pli.setLayoutCount = 1;
	pli.pSetLayouts = &m_temporalDescSetLayout;
	pli.pushConstantRangeCount = 1;
	pli.pPushConstantRanges = &pcRange;
	vkCreatePipelineLayout(m_device, &pli, nullptr, &m_temporalPipeLayout);

	// Compute pipeline
	std::vector<std::string> paths = defaultSearchPaths;
	auto spv = nvh::loadFile("spv/temporal_pass1.comp.spv", true, paths, true);
	VkShaderModule mod = nvvk::createShaderModule(m_device, spv);

	VkComputePipelineCreateInfo ci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
	ci.layout = m_temporalPipeLayout;
	ci.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
				VK_SHADER_STAGE_COMPUTE_BIT, mod, "main", nullptr };

	vkCreateComputePipelines(m_device, {}, 1, &ci, nullptr, &m_temporalPipeline);
	vkDestroyShaderModule(m_device, mod, nullptr);
}

void HelloVulkan::updateTemporalDescriptorSet(int readIdx, int writeIdx, uint32_t frameIdx)
{
	VkDescriptorSet ds = m_temporalDescSets[frameIdx];

	VkDescriptorImageInfo histPrev = m_histMoments[readIdx].descriptor;
	VkDescriptorImageInfo normalPrev = m_prevNormal.descriptor;
	VkDescriptorImageInfo depthPrev = m_prevDepth.descriptor;
	VkDescriptorImageInfo normalCur = m_offscreenNormal.descriptor;
	VkDescriptorImageInfo depthCur = m_offscreenLinearDepth.descriptor;
	VkDescriptorImageInfo motion = m_offscreenMotion.descriptor;
	VkDescriptorImageInfo noisy = m_offscreenNoisyShadow.descriptor; // your noisy shadow src

	// storage target must be GENERAL
	VkDescriptorImageInfo outMoments{ {}, m_histMoments[writeIdx].descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL };

	std::vector<VkWriteDescriptorSet> writes;
	writes.emplace_back(m_temporalDSBind.makeWrite(ds, 0, &histPrev));
	writes.emplace_back(m_temporalDSBind.makeWrite(ds, 1, &normalPrev));
	writes.emplace_back(m_temporalDSBind.makeWrite(ds, 2, &depthPrev));
	writes.emplace_back(m_temporalDSBind.makeWrite(ds, 3, &normalCur));
	writes.emplace_back(m_temporalDSBind.makeWrite(ds, 4, &depthCur));
	writes.emplace_back(m_temporalDSBind.makeWrite(ds, 5, &motion));
	writes.emplace_back(m_temporalDSBind.makeWrite(ds, 6, &noisy));
	writes.emplace_back(m_temporalDSBind.makeWrite(ds, 7, &outMoments));

	vkUpdateDescriptorSets(m_device, (uint32_t)writes.size(), writes.data(), 0, nullptr);
}

void HelloVulkan::runTemporalPass1(VkCommandBuffer cmd, uint32_t frameIdx)
{
	// Choose ping-pong indices
	int writeIdx = m_histIdx;
	int readIdx = 1 - m_histIdx;

	// Make RT writes visible to compute (normal, depth, motion, noisy)
	VkPipelineStageFlags src = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
	VkPipelineStageFlags dst = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

	std::vector<VkImageMemoryBarrier> ib;
	ib.push_back(makeRtToFsBarrier(m_offscreenNormal.image));
	ib.push_back(makeRtToFsBarrier(m_offscreenLinearDepth.image));
	ib.push_back(makeRtToFsBarrier(m_offscreenMotion.image));
	// Add barrier for your noisy shadow image too if different from color
	ib.push_back(makeRtToFsBarrier(m_offscreenNoisyShadow.image));

	if (!ib.empty()) {
		vkCmdPipelineBarrier(cmd, src, dst, 0, 0, nullptr, 0, nullptr,
			(uint32_t)ib.size(), ib.data());
	}

	// Update descriptors to use readIdx as prev, writeIdx as output
	updateTemporalDescriptorSet(readIdx, writeIdx, frameIdx);

	// bind this frame's set
	VkDescriptorSet ds = m_temporalDescSets[frameIdx];
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_temporalPipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_temporalPipeLayout,
		0, 1, &ds, 0, nullptr);

	struct Params { float tauZ, tauN, clampK, alphaUse; int firstFrame; } pc;
	pc.tauZ = 0.02f;         // tune
	pc.tauN = 0.97f;         // tune
	pc.clampK = 1.0f;        // tune
	pc.alphaUse = 0.1f;      // tune
	pc.firstFrame = m_firstFrame ? 1 : 0;

	vkCmdPushConstants(cmd, m_temporalPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Params), &pc);

	// Dispatch
	const uint32_t gx = (m_size.width + 7) / 8;
	const uint32_t gy = (m_size.height + 7) / 8;
	vkCmdDispatch(cmd, gx, gy, 1);

	// Compute -> (next consumer) barrier
	// If next you will sample the new moments in a post FS or a spatial compute pass:
	VkImageMemoryBarrier outBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
	outBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	outBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	outBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
	outBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	outBarrier.image = m_histMoments[writeIdx].image;
	outBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		0, 0, nullptr, 0, nullptr, 1, &outBarrier);

	// Publish current guides as "prev" for next frame
	// (You can keep your copy code here: normal -> prevNormal, depth -> prevDepth)
	{
		// to copy: src must be TRANSFER_SRC, dst TRANSFER_DST
		nvvk::cmdBarrierImageLayout(cmd, m_offscreenNormal.image,
			VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
		nvvk::cmdBarrierImageLayout(cmd, m_prevNormal.image,
			VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

		VkImageCopy reg{};
		reg.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
		reg.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
		reg.extent = { m_size.width, m_size.height, 1 };
		vkCmdCopyImage(cmd,
			m_offscreenNormal.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			m_prevNormal.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1, &reg);

		// restore to GENERAL for next use
		nvvk::cmdBarrierImageLayout(cmd, m_offscreenNormal.image,
			VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
		nvvk::cmdBarrierImageLayout(cmd, m_prevNormal.image,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
	}

	// Copy current linear depth -> prevDepth  (same pattern)
	{
		nvvk::cmdBarrierImageLayout(cmd, m_offscreenLinearDepth.image,
			VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
		nvvk::cmdBarrierImageLayout(cmd, m_prevDepth.image,
			VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

		VkImageCopy reg{};
		reg.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
		reg.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
		reg.extent = { m_size.width, m_size.height, 1 };
		vkCmdCopyImage(cmd,
			m_offscreenLinearDepth.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			m_prevDepth.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1, &reg);

		nvvk::cmdBarrierImageLayout(cmd, m_offscreenLinearDepth.image,
			VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
		nvvk::cmdBarrierImageLayout(cmd, m_prevDepth.image,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
	}

	// Flip ping pong
	m_histIdx = 1 - m_histIdx;
	m_firstFrame = false;
}

// --- create descriptors (once) ---
void HelloVulkan::createSpatialDescriptor()
{
	m_spatialDSBind = {};
	// inputs
	m_spatialDSBind.addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT); // meanVar_t
	m_spatialDSBind.addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT); // normal
	m_spatialDSBind.addBinding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT); // depth
	// output
	m_spatialDSBind.addBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);  // denoised

	m_spatialDescSetLayout = m_spatialDSBind.createLayout(m_device);
	m_spatialDescPool = m_spatialDSBind.createPool(m_device, m_framesInFlight);
	m_spatialDescSets.resize(m_framesInFlight);
	for (uint32_t i = 0; i < m_framesInFlight; ++i)
		m_spatialDescSets[i] = nvvk::allocateDescriptorSet(m_device, m_spatialDescPool, m_spatialDescSetLayout);
}

void HelloVulkan::createSpatialPipeline()
{
	struct PC { float invSigmaZ, invSigmaN; float varToRadius; float varClamp; };
	VkPushConstantRange pcRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC) };

	VkPipelineLayoutCreateInfo pli{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
	pli.setLayoutCount = 1; pli.pSetLayouts = &m_spatialDescSetLayout;
	pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcRange;
	vkCreatePipelineLayout(m_device, &pli, nullptr, &m_spatialPipeLayout);

	auto spv = nvh::loadFile("spv/spatial_pass2.comp.spv", true, defaultSearchPaths, true);
	VkShaderModule mod = nvvk::createShaderModule(m_device, spv);

	VkComputePipelineCreateInfo ci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
	ci.layout = m_spatialPipeLayout;
	ci.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
				  VK_SHADER_STAGE_COMPUTE_BIT, mod, "main", nullptr };
	vkCreateComputePipelines(m_device, {}, 1, &ci, nullptr, &m_spatialPipeline);
	vkDestroyShaderModule(m_device, mod, nullptr);
}

// Per-frame descriptor update (readIdx = the index Pass1 just wrote)
void HelloVulkan::updateSpatialDescriptorSet(int readIdx, uint32_t frameIdx)
{
	VkDescriptorSet ds = m_spatialDescSets[frameIdx];

	VkDescriptorImageInfo meanVar = m_histMoments[readIdx].descriptor;              // R16G16 -> (mean,var)
	VkDescriptorImageInfo normal = m_offscreenNormal.descriptor;
	VkDescriptorImageInfo depth = m_offscreenLinearDepth.descriptor;

	// output: choose/create an image for denoised shadows (e.g., VK_FORMAT_R16_SFLOAT)
	VkDescriptorImageInfo outDenoised{ {}, m_denoisedShadow.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL };

	std::vector<VkWriteDescriptorSet> writes;
	writes.emplace_back(m_spatialDSBind.makeWrite(ds, 0, &meanVar));
	writes.emplace_back(m_spatialDSBind.makeWrite(ds, 1, &normal));
	writes.emplace_back(m_spatialDSBind.makeWrite(ds, 2, &depth));
	writes.emplace_back(m_spatialDSBind.makeWrite(ds, 3, &outDenoised));
	vkUpdateDescriptorSets(m_device, (uint32_t)writes.size(), writes.data(), 0, nullptr);
}

void HelloVulkan::runSpatialPass2(VkCommandBuffer cmd, int readIdx, uint32_t frameIdx)
{
	// Make sure Pass 1 writes are visible to this compute
	VkImageMemoryBarrier ib{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
	ib.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; // Pass1 wrote mean/var
	ib.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;  // Pass2 reads it
	ib.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
	ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	ib.image = m_histMoments[readIdx].image;
	ib.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
	vkCmdPipelineBarrier(cmd,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,         // src = Pass1 compute
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,         // dst = this compute
		0, 0, nullptr, 0, nullptr, 1, &ib);

	updateSpatialDescriptorSet(readIdx, frameIdx);

	VkDescriptorSet ds = m_spatialDescSets[frameIdx];
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_spatialPipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_spatialPipeLayout, 0, 1, &ds, 0, nullptr);

	struct PC { float invSigmaZ, invSigmaN; float varToRadius; float varClamp; } pc;
	// Tunables (start here, tweak in UI later)
	//pc.invSigmaZ = 60.0f;  // stronger depth stop with larger value
	//pc.invSigmaN = 32.0f;  // strong normal stop (cos^32)
	//pc.varToRadius = 8.0f;   // map variance -> radius (0..~5)
	//pc.varClamp = 0.25f;  // optional clamp on max variance influence

	pc.invSigmaZ = m_spatialPC.invSigmaZ;
	pc.invSigmaN = m_spatialPC.invSigmaN;
	pc.varToRadius = m_spatialPC.varToRadius;
	pc.varClamp = m_spatialPC.varClamp;


	vkCmdPushConstants(cmd, m_spatialPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

	const uint32_t lx = 8, ly = 8;
	const uint32_t gx = (m_size.width + lx - 1) / lx;
	const uint32_t gy = (m_size.height + ly - 1) / ly;
	vkCmdDispatch(cmd, gx, gy, 1);

	// If you will sample m_denoisedShadow in your post pass, add a barrier:
	VkImageMemoryBarrier outB{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
	outB.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	outB.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	outB.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
	outB.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	outB.image = m_denoisedShadow.image;
	outB.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0,1,0,1 };
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		0, 0, nullptr, 0, nullptr, 1, &outB);
}



void HelloVulkan::onKeyboard(int key, int /*scancode*/, int action, int /*mods*/)
{
	if (action == GLFW_PRESS || action == GLFW_REPEAT)
	{
		float moveStep = 10000.0f;  // higher = faster movement
		switch (key)
		{
		case GLFW_KEY_W:
			CameraManip.keyMotion(0.0f, moveStep, 1);  // dolly forward
			break;
		case GLFW_KEY_S:
			CameraManip.keyMotion(0.0f, -moveStep, 1); // dolly backward
			break;
		case GLFW_KEY_A:
			CameraManip.keyMotion(-moveStep, 0.0f, 2); // pan left
			break;
		case GLFW_KEY_D:
			CameraManip.keyMotion(moveStep, 0.0f, 2);  // pan right
			break;
		}
	}
}
