#version 460

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;


layout(location = 0) out vec3 outColor;

layout(set = 0, binding = 0) uniform GlobalUniforms {
    mat4 viewProj;
    mat4 viewInverse;
    mat4 projInverse;
} ubo;

void main()
{
    gl_Position = ubo.viewProj * vec4(inPosition, 1.0);
    //gl_Position = vec4(inPosition.x / 2.0, inPosition.y / 2.0, 0.0, 1.0);
    outColor = inColor;
}
