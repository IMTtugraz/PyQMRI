#pragma OPENCL EXTENSION cl_khr_fp64: enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
void AtomicAdd(__global double *val, double delta)
{
    union
    {
        double f;
        ulong  i;
    } old;
    union
    {
        double f;
        ulong  i;
    } new;
    do
    {
        old.f = *val;
        new.f = old.f + delta;
    }
    while (atom_cmpxchg ( (volatile __global ulong *)val,
                          old.i, new.i) != old.i);
}


__kernel void make_complex(
                __global double2 *out,
                __global double *re,
                __global double* im
                )
{
    size_t k = get_global_id(0);

    out[k].s0 = re[k];
    out[k].s1 = im[k];
}


__kernel void deapo_adj(
                __global double2 *out,
                __global double2 *in,
                __constant double *deapo,
                const int dim,
                const double scale,
                const double ogf
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
                __global double2 *out,
                __global double2 *in,
                __constant double *deapo,
                const int dim,
                const double scale,
                const double ogf
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

__kernel void zero_tmp(__global double2 *tmp)
{
    size_t x = get_global_id(0);

    tmp[x] = 0.0f;
}

__kernel void grid_lut(
                __global double *sg,
                __global double2 *s,
                __global double2 *kpos,
                const int gridsize,
                const double kwidth,
                __global double *dcf,
                __constant double* kerneltable,
                const int nkernelpts,
                const int scanoffset
                )
{
    size_t k = get_global_id(2);
    size_t kDim = get_global_size(2);
    size_t n = get_global_id(1);
    size_t NDim = get_global_size(1);
    size_t scan = get_global_id(0);

    int ixmin, ixmax, iymin, iymax, gridcenter, gptr_cinc, kernelind,indx,indy;
    double kx, ky;
    double fracind, dkx, dky, dk, fracdk, kern;
    gridcenter = gridsize/2;

    double2 kdat = s[k+kDim*n+kDim*NDim*scan]*(double2)(dcf[k],dcf[k]);

    kx = (kpos[k+kDim*(scan+scanoffset)]).s0;
    ky = (kpos[k+kDim*(scan+scanoffset)]).s1;

    ixmin =  (int)((kx-kwidth)*gridsize +gridcenter);
    ixmax = (int)((kx+kwidth)*gridsize +gridcenter)+1;
    iymin = (int)((ky-kwidth)*gridsize +gridcenter);
    iymax =  (int)((ky+kwidth)*gridsize +gridcenter)+1;

    for (int gcount1 = ixmin; gcount1 <= ixmax; gcount1++)
    {
        dkx = (double)(gcount1-gridcenter) / (double)gridsize  - kx;
        for (int gcount2 = iymin; gcount2 <= iymax; gcount2++)
        {
            dky = (double)(gcount2-gridcenter) / (double)gridsize - ky;

            dk = sqrt(dkx*dkx+dky*dky);
            if (dk < kwidth)
            {
                fracind = dk/kwidth*(double)(nkernelpts-1);
                kernelind = (int)fracind;
                fracdk = fracind-(double)kernelind;

                kern = kerneltable[kernelind]*(1-fracdk)+
                kerneltable[kernelind+1]*fracdk;
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
          indx*gridsize+indy
          + (gridsize*gridsize)*n
          + (gridsize*gridsize)*NDim*scan)]
      ),
    (kern * kdat.s0));
                AtomicAdd(
    &(
      sg[
        2*(
          indx*gridsize+indy
          + (gridsize*gridsize)*n
          + (gridsize*gridsize)*NDim*scan)+1]
      ),
    (kern * kdat.s1));
            }
        }
    }
}


__kernel void invgrid_lut(
                __global double2 *s,
                __global double2 *sg,
                __global double2 *kpos,
                const int gridsize,
                const double kwidth,
                __global double *dcf,
                __constant double* kerneltable,
                const int nkernelpts,
                const int scanoffset
                )
{
    size_t k = get_global_id(2);
    size_t kDim = get_global_size(2);
    size_t n = get_global_id(1);
    size_t NDim = get_global_size(1);
    size_t scan = get_global_id(0);

    int ixmin, ixmax, iymin, iymax, gridcenter,gptr_cinc,kernelind, indx,indy;
    double kx, ky;
    double fracind, dkx, dky, dk, fracdk, kern;
    gridcenter = gridsize/2;

    double2 tmp_dat = 0.0;

    kx = (kpos[k+kDim*(scan+scanoffset)]).s0;
    ky = (kpos[k+kDim*(scan+scanoffset)]).s1;

    ixmin =  (int)((kx-kwidth)*gridsize +gridcenter);
    ixmax = (int)((kx+kwidth)*gridsize +gridcenter)+1;
    iymin = (int)((ky-kwidth)*gridsize +gridcenter);
    iymax =  (int)((ky+kwidth)*gridsize +gridcenter)+1;

    for (int gcount1 = ixmin; gcount1 <= ixmax; gcount1++)
    {
        dkx = (double)(gcount1-gridcenter) / (double)gridsize - kx;
        for (int gcount2 = iymin; gcount2 <= iymax; gcount2++)
        {
            dky = (double)(gcount2-gridcenter) / (double)gridsize - ky;

            dk = sqrt(dkx*dkx+dky*dky);

            if (dk < kwidth)
            {
                fracind = dk/kwidth*(double)(nkernelpts-1);
                kernelind = (int)fracind;
                fracdk = fracind-(double)kernelind;

                kern = kerneltable[kernelind]*(1-fracdk)+
                kerneltable[kernelind+1]*fracdk;
                indx = gcount1;
                indy = gcount2;
                if (gcount1 < 0) {indx+=gridsize;indy=gridsize-indy;}
                if (gcount1 >= gridsize) {indx-=gridsize;indy=gridsize-indy;}
                if (gcount2 < 0) {indy+=gridsize;indx=gridsize-indx;}
                if (gcount2 >= gridsize) {indy-=gridsize;indx=gridsize-indx;}

                tmp_dat += (double2)(kern,kern)*sg[
                                        indx*gridsize+indy
                                        + (gridsize*gridsize)*n
                                        + (gridsize*gridsize)*NDim*scan];
            }
        }
    }
    s[k+kDim*n+kDim*NDim*scan]= tmp_dat*(double2)(dcf[k],dcf[k]);
}


__kernel void copy(
                __global double2 *out,
                __global double2 *in,
                const double scale
                )
{
    size_t x = get_global_id(0);
    out[x] = in[x]*scale;
}


__kernel void copy_SMS_fwd(
                __global double2 *out,
                __global double2 *in,
                __global int* shift,
                const int packs,
                const int MB,
                const double scale,
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
              + idSlice*packs*gid
              + idSlice*packs*NGroups*n] = (double2)(0);
            for(int z=0; z< MB; z++)
            {
                out[
                x+y*dimX+idSlice*k
                + idSlice*packs*gid
                + idSlice*packs*NGroups*n] += in[
                        x+(y+shift[z])%dimY*dimX
                        + idSlice*(k+z*packs+gid*packs*MB)
                        + NSlice*n]*scale;
            }
        }
    }
}

__kernel void copy_SMS_adj(
                __global double2 *out,
                __global double2 *in,
                __global int* shift,
                const int packs,
                const int MB,
                const double scale,
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

__kernel void masking(__global double2 *ksp, __global double *mask)
{
    size_t x = get_global_id(0);
    ksp[x] = ksp[x]*mask[x];
}

__kernel void maskingcpy(
                __global double2* out,
                __global double2 *ksp,
                __global double *mask
                )
{
    size_t x = get_global_id(2);
    size_t dimX = get_global_size(2);
    size_t y = get_global_id(1);
    size_t dimY = get_global_size(1);
    size_t n = get_global_id(0);
    out[x+y*dimX+dimX*dimY*n] = ksp[x+y*dimX+dimX*dimY*n]*mask[x+y*dimX];
}

__kernel void fftshift(__global double2* ksp, __global double *check)
{
    size_t x = get_global_id(2);
    size_t dimX = get_global_size(2);
    size_t y = get_global_id(1);
    size_t dimY = get_global_size(1);
    size_t n = get_global_id(0);

    ksp[x+dimX*y+dimX*dimY*n] = ksp[x+dimX*y+dimX*dimY*n]*check[x]*check[y];
}

__kernel void copy_SMS_fwdkspace(
                __global double2 *out,
                __global double2 *in,
                __global double* shift,
                __global double *mask,
                const int packs,
                const int MB,
                const double scale,
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
    double expshift = 0.0f;
    double sinshift = 0.0f;
    double cosshift = 0.0f;
    int inind = 0;

    double exppos = ((double)x/(double)dimX-0.5f);
    double2 tmp = 0.0f;

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

                tmp += (double2) (in[inind].x*cosshift-in[inind].y*sinshift,
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
                __global double2 *out,
                __global double2 *in,
                __global double* shift,
                __global double *mask,
                const int packs,
                const int MB,
                const double scale,
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

    double expshift = 0.0f;
    double sinshift = 0.0f;
    double cosshift = 0.0f;
    int inind = 0;

    double exppos = ((double)x/(double)dimX-0.5f);

    double2 tmpin = 0.0f;

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
                    + NSlice*n] = (double2) (tmpin.x*cosshift-tmpin.y*sinshift,
                                            tmpin.x*sinshift+tmpin.y*cosshift);
            }
        }
    }
}