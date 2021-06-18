#pragma once

#if defined(naive)
    typedef unsigned short* rtype;
#else
    typedef unsigned * rtype;
#endif

void step_2(rtype hists_cpu, uint8_t* image, size_t width, size_t height);