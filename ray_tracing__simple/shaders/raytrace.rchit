#version 460
#extension GL_EXT_ray_tracing                          : require
#extension GL_EXT_nonuniform_qualifier                 : enable
#extension GL_EXT_scalar_block_layout                  : enable
#extension GL_GOOGLE_include_directive                 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2                    : require

#include "raycommon.glsl"
#include "wavefront.glsl"

layout(location = 0) rayPayloadInEXT hitPayload prd;
layout(location = 1) rayPayloadEXT bool isShadowed;

layout(buffer_reference, scalar) buffer Vertices   { Vertex v[];          };
layout(buffer_reference, scalar) buffer Indices    { ivec3  i[];          };
layout(buffer_reference, scalar) buffer Materials  { WaveFrontMaterial m[]; };
layout(buffer_reference, scalar) buffer MatIndices { int    i[];          };

layout(set = 0, binding = eTlas) uniform accelerationStructureEXT topLevelAS;

layout(set = 0, binding = eNormalImage, rgba16f) uniform image2D gNormal;
layout(set = 0, binding = eLinearDepth, r32f) uniform image2D gDepth;
layout(set = 0, binding = eMotionImage, rg16f) uniform image2D gMotion;
layout(set = 0, binding = eNoisyShadow, r16f) uniform image2D gShadowNoisy;

layout(set = 1, binding = eObjDescs, scalar) buffer ObjDesc_ { ObjDesc i[]; } objDesc;
layout(set = 1, binding = eTextures) uniform sampler2D textureSamplers[];
layout(set = 1, binding = eGlobals) uniform _GlobalUniforms { GlobalUniforms uni; };
layout(push_constant) uniform _PushConstantRay { PushConstantRay pcRay; };

hitAttributeEXT vec3 attribs;

const float EPS = 1e-3;
const float INF = 1e20;

// -----------------------------------------------------------------------------
// RNG helpers (simple hash-based; replace with a better RNG if you have one)
float rand(vec2 p) {
  return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

float rand(inout uint seed) {
  seed = (seed ^ 61u) ^ (seed >> 16);
  seed *= 9u;
  seed ^= (seed >> 4);
  seed *= 0x27d4eb2du;
  seed ^= (seed >> 15);
  return float(seed) * (1.0 / 4294967296.0);
}

// ---- integer hash (wang) ----------------------------------------------------
uint wang_hash(uint x)
{
    x = (x ^ 61u) ^ (x >> 16);
    x *= 9u;
    x ^= (x >> 4);
    x *= 0x27d4eb2du;
    x ^= (x >> 15);
    return x;
}

// ---- 32-bit to float in [0,1) ------------------------------------------------
float u32_to_float01(uint v)
{
    // divide by 2^32 as float
    return float(v) * (1.0 / 4294967296.0);
}

// ---- small xorshift-like step to advance rng state ---------------------------
uint rng_step(inout uint state)
{
    // A few ops to scramble state (cheap PRNG)
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    // mix with a constant
    state *= 0x9E3779B1u;
    state = wang_hash(state);
    return state;
}

// ---- convenience: per-pixel+per-frame random in [0,1) ------------------------
float randPixelFrame(ivec2 pixel, uint frame)
{
    // build a 32-bit seed from pixel coords and frame
    // combine coordinates in a way that reduces collisions
    uint seed = uint(pixel.x) + (uint(pixel.y) * 73856093u);    // pack pixel loc
    seed ^= (frame + 0x9e3779b9u);                             // xor with frame
    seed = wang_hash(seed);                                    // scramble
    // advance a couple times for decorrelation
    rng_step(seed);
    rng_step(seed);
    return u32_to_float01(seed);
}

// ---- alternate: seeded rand using existing surfKey/seed approach -------------
float randFromSeed(inout uint seed)
{
    uint v = rng_step(seed);
    return u32_to_float01(v);
}

// Shadow query wrapper. Returns true if any occluder is found.
bool traceShadowAnyHit(vec3 origin, vec3 dir, float tMax) {
  const uint flags =
      gl_RayFlagsTerminateOnFirstHitEXT |
      gl_RayFlagsOpaqueEXT |
      gl_RayFlagsSkipClosestHitShaderEXT;

  isShadowed = true;  // miss shader must set to false
  traceRayEXT(
      topLevelAS,
      flags,
      0xFF,
      0, 0, 1,          // sbtRecordOffset/Stride, missIndex
      origin,
      EPS,              // tMin
      dir,
      tMax,
      1                 // payload location = 1 (isShadowed)
  );
  return isShadowed;
}

// Sample a point on a rectangle light (center pcRay.lightPosition, edges U/V).
// Returns: sampled direction Ls, distance dist, cosine term at light cosThetaL, and the point.
void sampleRectLight(vec3 P, vec2 u, out vec3 Ls, out float dist, out float cosThetaL, out vec3 lightPt) {
  // Map [0,1)^2 to the rectangle
  lightPt = pcRay.lightPosition.xyz
          + (u.x - 0.5) * pcRay.lightU.xyz
          + (u.y - 0.5) * pcRay.lightV.xyz;

  vec3 lDir = lightPt - P;
  dist      = length(lDir);
  Ls        = lDir / max(dist, EPS);

  vec3 lightN = normalize(cross(pcRay.lightU.xyz, pcRay.lightV.xyz));
  cosThetaL   = max(0.0, dot(-Ls, lightN));
}

void main() {
  // ----- Fetch object resources -----
  ObjDesc    objRes      = objDesc.i[gl_InstanceCustomIndexEXT];
  MatIndices matIndices  = MatIndices(objRes.materialIndexAddress);
  Materials  materials   = Materials(objRes.materialAddress);
  Indices    indices     = Indices(objRes.indexAddress);
  Vertices   vertices    = Vertices(objRes.vertexAddress);

  // ----- Triangle data -----
  ivec3  tri  = indices.i[gl_PrimitiveID];
  Vertex v0   = vertices.v[tri.x];
  Vertex v1   = vertices.v[tri.y];
  Vertex v2   = vertices.v[tri.z];

  vec3  bary  = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
  vec3  Pobj  = v0.pos * bary.x + v1.pos * bary.y + v2.pos * bary.z;   // object space
  vec3  Nobj  = normalize(v0.nrm * bary.x + v1.nrm * bary.y + v2.nrm * bary.z);

  vec3  P     = vec3(gl_ObjectToWorldEXT * vec4(Pobj, 1.0));
  // world normal = (M^-1)^T * n, where M = objectToWorld. We have gl_WorldToObjectEXT = M^-1
  vec3  N     = normalize(mat3(transpose(gl_WorldToObjectEXT)) * Nobj);

  // Material (single lookup)
  int                 matIdx = matIndices.i[gl_PrimitiveID];
  WaveFrontMaterial   mat    = materials.m[matIdx];

  // Store World Normal
  vec3 V = normalize(-gl_WorldRayDirectionEXT); // view (from hit point toward camera) 
  vec3 Nf = normalize(faceforward(N, -V, N)); // make it consistently oriented 
  // Write raw [-1,1] in RGBA16F (good for ML/denoisers); or encode to [0,1] if you prefer to visualize. 
  imageStore(gNormal, ivec2(gl_LaunchIDEXT.xy), vec4(Nf, 1.0));

  // Store Linear Depth
  float d = gl_HitTEXT * length(gl_WorldRayDirectionEXT);                      // <- valid here, if ray dir is unit

  // closest-hit.glsl
  vec3 X = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT; // world hit
  vec3 camPos  = (uni.viewInverse * vec4(0,0,0,1)).xyz;
  vec3 camFwd  = normalize(-mat3(uni.viewInverse)[2]);  // camera forward (-Z row of viewInverse)
  float z_view = dot(X - camPos, camFwd);               // positive forward distance
  imageStore(gDepth, ivec2(gl_LaunchIDEXT.xy), vec4(z_view,0,0,0));
  //imageStore(gDepth, ivec2(gl_LaunchIDEXT.xy), vec4(d,0,0,0));

  // Motion vectors (pixels). viewProj == VP_curr per our UBO plan.
  vec4 c_curr = uni.viewProj * vec4(X, 1.0);
  vec4 c_prev = uni.VP_prev  * vec4(X, 1.0);

  // Guard against invalid projections (behind camera etc.)
  bool ok_curr = abs(c_curr.w) > 1e-6;
  bool ok_prev = abs(c_prev.w) > 1e-6;

  vec2 motion_px = vec2(0.0);
  if (ok_curr && ok_prev) {
      vec2 ndc_curr = c_curr.xy / c_curr.w;   // [-1,1]
      vec2 ndc_prev = c_prev.xy / c_prev.w;
      motion_px = 0.5 * (ndc_curr - ndc_prev) * uni.viewportSize; // pixels
  }

  // Write motion (RG). You can pack to half; sign is fine.
  imageStore(gMotion, ivec2(gl_LaunchIDEXT.xy), vec4(motion_px, 0.0, 0.0)); 

  // Optional texture
  vec3 albedoTex = vec3(1.0);
  if (mat.textureId >= 0) {
    uint txtId   = uint(mat.textureId) + objDesc.i[gl_InstanceCustomIndexEXT].txtOffset;
    vec2 uv      = v0.texCoord * bary.x + v1.texCoord * bary.y + v2.texCoord * bary.z;
    albedoTex    = texture(textureSamplers[nonuniformEXT(txtId)], uv).xyz;
  }

  // ---------------------------------------------------------------------------
  // Lighting
  // ---------------------------------------------------------------------------
  vec3  L        = vec3(0.0);
  float distMax  = INF;
  float lightI   = pcRay.lightIntensity;

  if (pcRay.lightType == 0) {
    // Point light
    vec3 lDir   = pcRay.lightPosition.xyz - P;
    float d     = length(lDir);
    L           = lDir / max(d, EPS);
    distMax     = d;
    lightI      = pcRay.lightIntensity / max(d * d, EPS);
  } else if (pcRay.lightType == 1) {
    // Directional light: direction stored in lightPosition (normalized preferred)
    L           = normalize(pcRay.lightPosition.xyz);
    distMax     = INF;
    lightI      = pcRay.lightIntensity;
  } else { // pcRay.lightType == 2  (rectangle area light)
    const int   NUM_SAMPLES = 1;
    vec3        accum       = vec3(0.0);
    int unoccluded = 0;

    // Precompute view dir for specular (keep your convention)
    vec3 V = gl_WorldRayDirectionEXT;

    for (int i = 0; i < NUM_SAMPLES; ++i) {
      // Low-discrepancy-ish jitter
      vec2 u = vec2(fract(rand(P.xy + vec2(float(i), 0.0))),
                    fract(rand(P.yz + vec2(0.0, float(i)))));

      // get pixel coords inside closest-hit: gl_LaunchIDEXT.xy
      //ivec2 px = ivec2(gl_LaunchIDEXT.xy);
      //
      //// if you have frame index in Uniforms (see below), use it:
      //uint frame = uint(uni.frameIndex); // add this field (see note)
      //
      //// produce 1 (or many) random samples
      //vec2 u;
      //u.x = randPixelFrame(px, frame);
      //u.y = randPixelFrame(px + ivec2(17, 29), frame); // offset to decorrelate components
      
      vec3 Ls, lightPt;
      float d, cosThetaL;
      sampleRectLight(P, u, Ls, d, cosThetaL, lightPt);

      // Intensity model: I * (cos at light) * area / r^2
      float I = pcRay.lightIntensity * cosThetaL * pcRay.lightArea / max(d * d, EPS);

      bool blocked = traceShadowAnyHit(P + N * EPS, Ls, d - EPS);
      float shadow = blocked ? 0.0001 : 1.0;
      unoccluded += blocked ? 0 : 1;

      vec3  diff = computeDiffuse(mat, Ls, N) * albedoTex;
      vec3  spec = blocked ? vec3(0.0) : computeSpecular(mat, V, Ls, N);

      accum += shadow * I * (diff + spec);
    }

    float visibility = float(unoccluded) / float(NUM_SAMPLES);
    imageStore(gShadowNoisy, ivec2(gl_LaunchIDEXT.xy), vec4(visibility,0,0,0));

    prd.hitValue = accum / float(NUM_SAMPLES);
    return;
  }

  // Point/Directional path
  // Only consider if light is on the front side
  vec3 color = vec3(0.0);
  float visibility = 0.0;
  if (dot(N, L) > 0.0) {
    bool blocked = traceShadowAnyHit(P + N * EPS, L, distMax - EPS);
    float shadow = blocked ? 0.0001 : 1.0;

    vec3 diff = computeDiffuse(mat, L, N) * albedoTex;
    vec3 spec = blocked ? vec3(0.0) : computeSpecular(mat, gl_WorldRayDirectionEXT, L, N);

    color = lightI * shadow * (diff + spec);
    visibility = blocked ? 0.0 : 1.0;
  }
  imageStore(gShadowNoisy, ivec2(gl_LaunchIDEXT.xy), vec4(visibility,0,0,0));

  prd.hitValue = color;
}
