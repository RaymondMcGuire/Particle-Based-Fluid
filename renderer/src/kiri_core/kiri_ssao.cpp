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

    mSSAOFBO = mSSAOBlurFBO = 0;
    mSSAOColorBuffer = mSSAOColorBufferBlur = 0;

    mSSAO = NULL;
    mSSAOBlur = NULL;
    mQuad = NULL;
}

void KiriSSAO::Enable()
{
    glGenFramebuffers(1, &mSSAOFBO);
    glGenFramebuffers(1, &mSSAOBlurFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, mSSAOFBO);

    // mSSAO color buffer
    glGenTextures(1, &mSSAOColorBuffer);
    glBindTexture(GL_TEXTURE_2D, mSSAOColorBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mSSAOColorBuffer, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "mSSAO Framebuffer not complete!" << std::endl;
    // and Blur stage
    glBindFramebuffer(GL_FRAMEBUFFER, mSSAOBlurFBO);
    glGenTextures(1, &mSSAOColorBufferBlur);
    glBindTexture(GL_TEXTURE_2D, mSSAOColorBufferBlur);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mSSAOColorBufferBlur, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "mSSAO Blur Framebuffer not complete!" << std::endl;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // generate mSSAO mKernel
    SampleKernel();
    // generate noise texture
    GenerateNoiseTexure();

    mSSAO = std::make_shared<KiriMaterialSSAO>(mSSAOKernel);
    mSSAOBlur = std::make_shared<KiriMaterialSSAOBlur>();
    mQuad = std::make_shared<KiriQuad>();
}

void KiriSSAO::InitSSAO(UInt mGPosition, UInt mGNormal)
{
    mQuad->SetMaterial(mSSAO);
    glBindFramebuffer(GL_FRAMEBUFFER, mSSAOFBO);
    glClear(GL_COLOR_BUFFER_BIT);
    mQuad->BindShader();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mGPosition);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, mGNormal);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, mNoiseTexture);
    mQuad->Draw();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void KiriSSAO::Blur()
{
    mQuad->SetMaterial(mSSAOBlur);
    glBindFramebuffer(GL_FRAMEBUFFER, mSSAOBlurFBO);
    glClear(GL_COLOR_BUFFER_BIT);
    mQuad->BindShader();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mSSAOColorBuffer);
    mQuad->Draw();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void KiriSSAO::Render(UInt mGPosition, UInt mGNormal)
{
    InitSSAO(mGPosition, mGNormal);
    Blur();
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

void KiriSSAO::SampleKernel()
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
        mSSAOKernel.append(sample);
    }
}

void KiriSSAO::GenerateNoiseTexure()
{

    for (UInt i = 0; i < 16; i++)
    {
        Vector3F noise(random<float>(0.0f, 1.0f) * 2.0f - 1.0f, random<float>(0.0f, 1.0f) * 2.0f - 1.0f, 0.0f); // Rotate around z-axis (in tangent space)
        mSSAONoise.append(noise);
    }

    glGenTextures(1, &mNoiseTexture);
    glBindTexture(GL_TEXTURE_2D, mNoiseTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, 4, 4, 0, GL_RGB, GL_FLOAT, &mSSAONoise[0]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
}

KiriSSAO::~KiriSSAO()
{
}
