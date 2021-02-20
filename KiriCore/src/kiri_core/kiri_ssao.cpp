/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-20 19:15:10 
 * @Last Modified by:   Xu.Wang 
 * @Last Modified time: 2020-03-20 19:15:10 
 */
#include <kiri_core/kiri_ssao.h>

KiriSSAO::KiriSSAO(UInt _w, UInt _h)
{
    WINDOW_WIDTH = _w;
    WINDOW_HEIGHT = _h;

    ssaoFBO = ssaoBlurFBO = 0;
    ssaoColorBuffer = ssaoColorBufferBlur = 0;

    mSSAO = NULL;
    mSSAOBlur = NULL;
    quad = NULL;
}

void KiriSSAO::enable()
{
    glGenFramebuffers(1, &ssaoFBO);
    glGenFramebuffers(1, &ssaoBlurFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, ssaoFBO);

    // SSAO color buffer
    glGenTextures(1, &ssaoColorBuffer);
    glBindTexture(GL_TEXTURE_2D, ssaoColorBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ssaoColorBuffer, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "SSAO Framebuffer not complete!" << std::endl;
    // and blur stage
    glBindFramebuffer(GL_FRAMEBUFFER, ssaoBlurFBO);
    glGenTextures(1, &ssaoColorBufferBlur);
    glBindTexture(GL_TEXTURE_2D, ssaoColorBufferBlur);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ssaoColorBufferBlur, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "SSAO Blur Framebuffer not complete!" << std::endl;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    //generate ssao mKernel
    sampleKernel();
    //generate noise texture
    generateNoiseTexure();

    mSSAO = std::make_shared<KiriMaterialSSAO>(ssaoKernel);
    mSSAOBlur = std::make_shared<KiriMaterialSSAOBlur>();
    quad = std::make_shared<KiriQuad>();
}

void KiriSSAO::ssao(UInt gPosition, UInt gNormal)
{
    quad->SetMaterial(mSSAO);
    glBindFramebuffer(GL_FRAMEBUFFER, ssaoFBO);
    glClear(GL_COLOR_BUFFER_BIT);
    quad->BindShader();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, gPosition);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, gNormal);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, noiseTexture);
    quad->Draw();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void KiriSSAO::blur()
{
    quad->SetMaterial(mSSAOBlur);
    glBindFramebuffer(GL_FRAMEBUFFER, ssaoBlurFBO);
    glClear(GL_COLOR_BUFFER_BIT);
    quad->BindShader();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, ssaoColorBuffer);
    quad->Draw();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void KiriSSAO::render(UInt gPosition, UInt gNormal)
{
    ssao(gPosition, gNormal);
    blur();
}

float lerp(float a, float b, float f)
{
    return a + f * (b - a);
}

std::default_random_engine generator;
template <typename T>
T random(float a, float b)
{
    // generates random floats between 0.0 and 1.0
    std::uniform_real_distribution<float> randomFloats(a, b);
    return randomFloats(generator);
}

void KiriSSAO::sampleKernel()
{
    for (UInt i = 0; i < 64; ++i)
    {
        Vector3F sample(random<float>(0.0f, 1.0f) * 2.0f - 1.0f, random<float>(0.0f, 1.0f) * 2.0f - 1.0f, random<float>(0.0f, 1.0f));
        sample.normalize();
        sample *= random<float>(0.0f, 1.0f);
        float Scale = float(i) / 64.0f;

        // Scale samples s.t. they're more aligned to center of mKernel
        Scale = lerp(0.1f, 1.0f, Scale * Scale);
        sample *= Scale;
        ssaoKernel.append(sample);
    }
}

void KiriSSAO::generateNoiseTexure()
{

    for (UInt i = 0; i < 16; i++)
    {
        Vector3F noise(random<float>(0.0f, 1.0f) * 2.0f - 1.0f, random<float>(0.0f, 1.0f) * 2.0f - 1.0f, 0.0f); // Rotate around z-axis (in tangent space)
        ssaoNoise.append(noise);
    }

    glGenTextures(1, &noiseTexture);
    glBindTexture(GL_TEXTURE_2D, noiseTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, 4, 4, 0, GL_RGB, GL_FLOAT, &ssaoNoise[0]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
}

KiriSSAO::~KiriSSAO()
{
}
