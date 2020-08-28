void AtomicAdd(volatile __global float *addr, float val)
{
    union
    {
        unsigned int u32;
        float        f32;
    } next, expected, current;
    current.f32    = *addr;
    do
    {
        expected.f32 = current.f32;
        next.f32     = expected.f32 + val;
        current.u32  = atomic_cmpxchg((volatile __global unsigned int *)addr,
                                      expected.u32, next.u32);
    }
    while( current.u32 != expected.u32 );
}

__kernel void make_complex(
                __global float2 *out,
                __global float *re,
                __global float* im
                )
{
    size_t k = get_global_id(0);

    out[k].s0 = re[k];
    out[k].s1 = im[k];
}


__kernel void deapo_adj(
                __global float2 *out,
                __global float2 *in,
                __constant float *deapo,
                const int dim,
                const float scale,
                const float ogf
                )
{
    size_t x = get_global_id(2);
    size_t X = get_global_size(2);
    size_t y = get_global_id(1);
    size_t Y = get_global_size(1);
    size_t k = get_global_id(0);

    size_t m = x+(int)(dim-X)/2;
    size_t M = dim;
    size_t n = y+(int)(dim-Y)/2;
    size_t N = dim;

    out[k*X*Y+y*X+x] = in[k*N*M+n*M+m] * deapo[y]* deapo[x] * scale;
}


__kernel void deapo_fwd(
                __global float2 *out,
                __global float2 *in,
                __constant float *deapo,
                const int dim,
                const float scale,
                const float ogf
                )
{
    size_t x = get_global_id(2);
    size_t X = get_global_size(2);
    size_t y = get_global_id(1);
    size_t Y = get_global_size(1);
    size_t k = get_global_id(0);

    size_t m = x+(int)(dim-X)/2;
    size_t M = dim;
    size_t n = y+(int)(dim-Y)/2;
    size_t N = dim;


    out[k*N*M+n*M+m] = in[k*X*Y+y*X+x] * deapo[y]* deapo[x] * scale;
}

__kernel void zero_tmp(__global float2 *tmp)
{
    size_t x = get_global_id(0);

    tmp[x] = 0.0f;
}

__kernel void grid_lut(
                __global float *sg,
                __global float2 *s,
                __global float2 *kpos,
                const int gridsize,
                const int NC,
                const float kwidth,
                __global float *dcf,
                __constant float* kerneltable,
                const int nkernelpts )
{
    size_t k = get_global_id(2);
    size_t kDim = get_global_size(2);
    size_t n = get_global_id(1);
    size_t NDim = get_global_size(1);
    size_t slice = get_global_id(0);

    size_t scan = n/NC;
    size_t icoil = n-scan*NC;

    int ixmin, ixmax, iymin, iymax, gridcenter, gptr_cinc,kernelind, indx,indy;
    float kx, ky;
    float fracind, dkx, dky, dk, fracdk, kern;
    gridcenter = gridsize/2;

    float2 kdat = s[k+kDim*n+kDim*NDim*slice]*(float2)(dcf[k],dcf[k]);

    kx = (kpos[k+kDim*scan]).s0;
    ky = (kpos[k+kDim*scan]).s1;

    ixmin =  (int)((kx-kwidth)*gridsize +gridcenter);
    ixmax = (int)((kx+kwidth)*gridsize +gridcenter)+1;
    iymin = (int)((ky-kwidth)*gridsize +gridcenter);
    iymax =  (int)((ky+kwidth)*gridsize +gridcenter)+1;

    for (int gcount1 = ixmin; gcount1 <= ixmax; gcount1++)
    {
        dkx = (float)(gcount1-gridcenter) / (float)gridsize - kx;
        for (int gcount2 = iymin; gcount2 <= iymax; gcount2++)
        {
            dky = (float)(gcount2-gridcenter) / (float)gridsize - ky;

            dk = sqrt(dkx*dkx+dky*dky);

            if (dk < kwidth)
            {
                fracind = dk/kwidth*(float)(nkernelpts-1);
                kernelind = (int)fracind;
                fracdk = fracind-(float)kernelind;

                kern = kerneltable[(int)kernelind]*(1-fracdk)+
                kerneltable[(int)kernelind+1]*fracdk;
                indx = gcount1;
                indy = gcount2;
                if (gcount1 < 0) {indx+=gridsize;indy=gridsize-indy;}
                if (gcount1 >= gridsize) {indx-=gridsize;indy=gridsize-indy;}
                if (gcount2 < 0) {indy+=gridsize;indx=gridsize-indx;}
                if (gcount2 >= gridsize) {indy-=gridsize;indx=gridsize-indx;}
                AtomicAdd(
    &(
      sg[
        2*(
          indx*gridsize+indy+(gridsize*gridsize)*n
          + (gridsize*gridsize)*NDim*slice)]
      ),
    (kern * kdat.s0));
                AtomicAdd(
    &(
      sg[
        2*(
          indx*gridsize+indy+(gridsize*gridsize)*n
          + (gridsize*gridsize)*NDim*slice)+1]
      ),
    (kern * kdat.s1));
            }
        }
    }
}


__kernel void invgrid_lut(
                __global float2 *s,
                __global float2 *sg,
                __global float2 *kpos,
                const int gridsize,
                const int NC,
                const float kwidth,
                __global float *dcf,
                __constant float* kerneltable,
                const int nkernelpts
                )
{
    size_t k = get_global_id(2);
    size_t kDim = get_global_size(2);
    size_t n = get_global_id(1);
    size_t NDim = get_global_size(1);
    size_t slice = get_global_id(0);

    size_t scan = n/NC;
    size_t icoil = n-scan*NC;

    int ixmin, ixmax, iymin, iymax, gridcenter, gptr_cinc,kernelind,indx,indy;
    float kx, ky;
    float fracind, dkx, dky, dk, fracdk, kern;
    gridcenter = gridsize/2;

    float2 tmp_dat = 0.0;


    kx = (kpos[k+kDim*scan]).s0;
    ky = (kpos[k+kDim*scan]).s1;

    ixmin =  (int)((kx-kwidth)*gridsize +gridcenter);
    ixmax = (int)((kx+kwidth)*gridsize +gridcenter)+1;
    iymin = (int)((ky-kwidth)*gridsize +gridcenter);
    iymax =  (int)((ky+kwidth)*gridsize +gridcenter)+1;

    for (int gcount1 = ixmin; gcount1 <= ixmax; gcount1++)
    {
        dkx = (float)(gcount1-gridcenter) / (float)gridsize  - kx;
        for (int gcount2 = iymin; gcount2 <= iymax; gcount2++)
        {
            dky = (float)(gcount2-gridcenter) / (float)gridsize - ky;

            dk = sqrt(dkx*dkx+dky*dky);

            if (dk < kwidth)
            {

                fracind = dk/kwidth*(float)(nkernelpts-1);
                kernelind = (int)fracind;
                fracdk = fracind-(float)kernelind;

                kern = kerneltable[(int)kernelind]*(1-fracdk)+
                kerneltable[(int)kernelind+1]*fracdk;
                indx = gcount1;
                indy = gcount2;
                if (gcount1 < 0) {indx+=gridsize;indy=gridsize-indy;}
                if (gcount1 >= gridsize) {indx-=gridsize;indy=gridsize-indy;}
                if (gcount2 < 0) {indy+=gridsize;indx=gridsize-indx;}
                if (gcount2 >= gridsize) {indy-=gridsize;indx=gridsize-indx;}
                tmp_dat += (float2)(kern,kern)*sg[
                                        indx*gridsize+indy
                                        + (gridsize*gridsize)*n
                                        + (gridsize*gridsize)*NDim*slice];
            }
        }
    }
    s[k+kDim*n+kDim*NDim*slice]= tmp_dat*(float2)(dcf[k],dcf[k]);
}


__kernel void copy(
                __global float2 *out,
                __global float2 *in,
                const float scale
                )
{
    size_t x = get_global_id(0);
    out[x] = in[x]*scale;
}


__kernel void copy_SMS_fwd(
                __global float2 *out,
                __global float2 *in,
                __global int* shift,
                const int packs,
                const int MB,
                const float scale,
                const int NGroups
                )
{
    size_t x = get_global_id(2);
    size_t dimX = get_global_size(2);
    size_t y = get_global_id(1);
    size_t dimY = get_global_size(1);
    size_t n = get_global_id(0);

    size_t NSlice = NGroups*packs*MB*dimX*dimY;
    size_t idSlice = dimX*dimY;

    for (int gid=0; gid<NGroups; gid++)
    {
        for (int k=0; k<packs; k++)
        {
            out[
              x+y*dimX+idSlice*k
              + idSlice*packs*gid+idSlice*packs*NGroups*n] = (float2)(0);
            for(int z=0; z< MB; z++)
            {
                out[
                  x+y*dimX+idSlice*k
                  + idSlice*packs*gid+idSlice*packs*NGroups*n] += in[
                              x+(y+shift[z])%dimY*dimX
                              + idSlice*(k+z*packs+gid*packs*MB)
                              + NSlice*n]*scale;
            }
        }
    }
}

__kernel void copy_SMS_adj(
                __global float2 *out,
                __global float2 *in,
                __global int* shift,
                const int packs,
                const int MB,
                const float scale,
                const int NGroups
                )
{
    size_t x = get_global_id(2);
    size_t dimX = get_global_size(2);
    size_t y = get_global_id(1);
    size_t dimY = get_global_size(1);
    size_t n = get_global_id(0);

    size_t NSlice = NGroups*packs*MB*dimX*dimY;
    size_t idSlice = dimX*dimY;

    for (int gid=0; gid<NGroups; gid++)
    {
        for (int k=0; k<packs; k++)
        {
            for(int z=0; z<MB; z++)
            {
                out[
                    x+(y+shift[z])%dimY*dimX
                    + idSlice*(k+z*packs+gid*packs*MB)
                    + NSlice*n] = in[
                            x+y*dimX+idSlice*k
                            + idSlice*packs*gid
                            + idSlice*packs*NGroups*n]*scale;
            }
        }
    }
}


__kernel void masking(__global float2 *ksp, __global float *mask)
{
    size_t x = get_global_id(0);
    ksp[x] = ksp[x]*mask[x];
}


__kernel void maskingcpy(
                __global float2* out,
                __global float2 *ksp,
                __global float *mask
                )
{
    size_t x = get_global_id(2);
    size_t dimX = get_global_size(2);
    size_t y = get_global_id(1);
    size_t dimY = get_global_size(1);
    size_t n = get_global_id(0);
    out[x+y*dimX+dimX*dimY*n] = ksp[x+y*dimX+dimX*dimY*n]*mask[x+y*dimX];
}


__kernel void fftshift(__global float2* ksp, __global float *check)
{
    size_t x = get_global_id(2);
    size_t dimX = get_global_size(2);
    size_t y = get_global_id(1);
    size_t dimY = get_global_size(1);
    size_t n = get_global_id(0);

    ksp[x+dimX*y+dimX*dimY*n] = ksp[x+dimX*y+dimX*dimY*n]*check[x]*check[y];
}


__kernel void copy_SMS_fwdkspace(
                __global float2 *out,
                __global float2 *in,
                __global float* shift,
                __global float *mask,
                const int packs,
                const int MB,
                const float scale,
                const int NGroups
                )
{
    size_t x = get_global_id(2);
    size_t dimX = get_global_size(2);
    size_t y = get_global_id(1);
    size_t dimY = get_global_size(1);
    size_t n = get_global_id(0);

    size_t NSlice = NGroups*packs*MB*dimX*dimY;
    size_t idSlice = dimX*dimY;
    float expshift = 0.0f;
    float sinshift = 0.0f;
    float cosshift = 0.0f;
    int inind = 0;

    float exppos = ((float)x/(float)dimX-0.5f);
    float2 tmp = 0.0f;

    for (int gid=0; gid<NGroups; gid++)
    {
        for (int k=0; k<packs; k++)
        {
            tmp = 0.0f;
            for(int z=0; z< MB; z++)
            {
                expshift = -2.0f*M_PI*shift[z]*exppos;
                sinshift = sin(expshift);
                cosshift = cos(expshift);

                inind = x+y*dimX+idSlice*(k+z*packs+gid*packs*MB)+NSlice*n;

                tmp += (float2) (in[inind].x*cosshift-in[inind].y*sinshift,
                                 in[inind].x*sinshift+in[inind].y*cosshift);
            }
            out[
                x+y*dimX+idSlice*k
                + idSlice*packs*gid
                + idSlice*packs*NGroups*n] = tmp*mask[x+y*dimX];
        }
    }
}


__kernel void copy_SMS_adjkspace(
                __global float2 *out,
                __global float2 *in,
                __global float* shift,
                __global float *mask,
                const int packs,
                const int MB,
                const float scale,
                const int NGroups
                )
{
    size_t x = get_global_id(2);
    size_t dimX = get_global_size(2);
    size_t y = get_global_id(1);
    size_t dimY = get_global_size(1);
    size_t n = get_global_id(0);

    size_t NSlice = NGroups*packs*MB*dimX*dimY;
    size_t idSlice = dimX*dimY;

    float expshift = 0.0f;
    float sinshift = 0.0f;
    float cosshift = 0.0f;
    int inind = 0;

    float exppos = ((float)x/(float)dimX-0.5f);

    float2 tmpin = 0.0f;

    for (int gid=0; gid<NGroups; gid++)
    {
        for (int k=0; k<packs; k++)
        {
            inind = x+y*dimX+idSlice*k
                    + idSlice*packs*gid+idSlice*packs*NGroups*n;
            tmpin = in[inind]*mask[x+y*dimX];
            for(int z=0; z<MB; z++)
            {
                expshift = 2.0f*M_PI*shift[z]*exppos;
                sinshift = sin(expshift);
                cosshift = cos(expshift);
                out[
                    x+y*dimX
                    + idSlice*(k+z*packs+gid*packs*MB)
                    + NSlice*n] = (float2) (tmpin.x*cosshift-tmpin.y*sinshift,
                                            tmpin.x*sinshift+tmpin.y*cosshift);
            }
        }
    }
}