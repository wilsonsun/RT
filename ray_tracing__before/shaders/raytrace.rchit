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

layout(set = 1, binding = eObjDescs, scalar) buffer ObjDesc_ { ObjDesc i[]; } objDesc;
layout(set = 1, binding = eTextures) uniform sampler2D textureSamplers[];

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
    const int   NUM_SAMPLES = 10;
    vec3        accum       = vec3(0.0);

    // Precompute view dir for specular (keep your convention)
    vec3 V = gl_WorldRayDirectionEXT;

    for (int i = 0; i < NUM_SAMPLES; ++i) {
      // Low-discrepancy-ish jitter
      vec2 u = vec2(fract(rand(P.xy + vec2(float(i), 0.0))),
                    fract(rand(P.yz + vec2(0.0, float(i)))));

      vec3 Ls, lightPt;
      float d, cosThetaL;
      sampleRectLight(P, u, Ls, d, cosThetaL, lightPt);

      // Intensity model: I * (cos at light) * area / r^2
      float I = pcRay.lightIntensity * cosThetaL * pcRay.lightArea / max(d * d, EPS);

      bool blocked = traceShadowAnyHit(P + N * EPS, Ls, d - EPS);
      float shadow = blocked ? 0.0001 : 1.0;

      vec3  diff = computeDiffuse(mat, Ls, N) * albedoTex;
      vec3  spec = blocked ? vec3(0.0) : computeSpecular(mat, V, Ls, N);

      accum += shadow * I * (diff + spec);
    }

    prd.hitValue = accum / float(NUM_SAMPLES);
    return;
  }

  // Point/Directional path
  // Only consider if light is on the front side
  vec3 color = vec3(0.0);
  if (dot(N, L) > 0.0) {
    bool blocked = traceShadowAnyHit(P + N * EPS, L, distMax - EPS);
    float shadow = blocked ? 0.0001 : 1.0;

    vec3 diff = computeDiffuse(mat, L, N) * albedoTex;
    vec3 spec = blocked ? vec3(0.0) : computeSpecular(mat, gl_WorldRayDirectionEXT, L, N);

    color = lightI * shadow * (diff + spec);
  }

  prd.hitValue = color;
}
