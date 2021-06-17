#pragma once

#if defined(naive)
    typedef unsigned short* rtype;
#else
    typedef unsigned * rtype;
#endif

void step_2(rtype hists_cpu, int nb_tiles_x, int nb_tiles_y);