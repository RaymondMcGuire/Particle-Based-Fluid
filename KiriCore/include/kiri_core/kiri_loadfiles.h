/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:41:00
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\kiri_loadfiles.h
 */

#ifndef _KIRI_LOAD_FILES_H_
#define _KIRI_LOAD_FILES_H_
#pragma once
#include <kiri_pch.h>
#include <kiri_define.h>
#include <root_directory.h>

class KiriLoadFiles
{
private:
    typedef String (*Builder)(const String &path);

public:
    static String getPath(const String &path)
    {
        static String (*pathBuilder)(String const &) = getPathBuilder();
        return (*pathBuilder)(path);
    }

    static String getBuildPath(const String &path)
    {
        static String (*pathBuilder)(String const &) = getBuildPathBuilder();
        return (*pathBuilder)(path);
    }

private:
    static String const &getRoot()
    {
#ifdef KIRI_WINDOWS
        static char *envRoot = nullptr;
        static char const *givenRoot = nullptr;
        size_t sz = 0;
        if (_dupenv_s(&envRoot, &sz, "ROOT_PATH_PATH") == 0 && envRoot != nullptr)
        {
            givenRoot = envRoot;
            free(envRoot);
        }
        else
        {
            givenRoot = ROOT_PATH;
        }
        static String root = (givenRoot != nullptr ? givenRoot : "");
        return root;
#endif

#ifdef KIRI_APPLE
        static char const *envRoot = getenv("ROOT_PATH_PATH");
        static char const *givenRoot = (envRoot != nullptr ? envRoot : ROOT_PATH);
        static String root = (givenRoot != nullptr ? givenRoot : "");
        return root;
#endif
    }

    static String const &getBuildRoot()
    {
#ifdef KIRI_WINDOWS
        static String root = ".";
        return root;
#endif

#ifdef KIRI_APPLE
        static char const *envRoot = getenv("LOGL_BUILD_ROOT_PATH");
        static char const *givenRoot = (envRoot != nullptr ? envRoot : MSWIN_BUILD_PATH);
        static String root = (givenRoot != nullptr ? givenRoot : "");
        return root;
#endif
    }

    static Builder getPathBuilder()
    {
        return &KiriLoadFiles::getPathRelativeRoot;
    }

    static Builder getBuildPathBuilder()
    {
        return &KiriLoadFiles::getBuildPathRelativeRoot;
    }

    static String getPathRelativeRoot(const String &path)
    {
        return getRoot() + String("/") + path;
    }

    static String getBuildPathRelativeRoot(const String &path)
    {
        return getBuildRoot() + String("/") + path;
    }
};

#endif
