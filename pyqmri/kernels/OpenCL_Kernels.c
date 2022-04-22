inline float2 cmult(float2 a, float2 b)
{
    return (float2)(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x);
}

inline float2 cmult_conj(float2 a, float2 b)
{
    return (float2)(a.x*b.x+a.y*b.y, -a.x*b.y+a.y*b.x);
}

__kernel void squarematvecmult(__global float2* outvec,
                          __global float2* mat,
                          __global float2* invec,
                          const int vecdim
                          )
{
    size_t Nx = get_global_size(2), Ny = get_global_size(1);
    size_t NSl = get_global_size(0);
    size_t x = get_global_id(2), y = get_global_id(1);
    size_t k = get_global_id(0);
    
    size_t ind_vec_out = k*Nx*Ny+Nx*y + x;
    size_t ind_vec_in = k*Nx*Ny+Nx*y + x;
    size_t ind_mat = k*Nx*Ny+Nx*y + x;
    
    for (int dim_x=0; dim_x<vecdim; dim_x++)
    {
        outvec[ind_vec_out] = 0.0f;
        ind_vec_in = k*Nx*Ny+Nx*y + x;
        for (int dim_y=0; dim_y<vecdim; dim_y++)
        {
            outvec[ind_vec_out] += cmult(invec[ind_vec_in], mat[ind_mat]);
            ind_vec_in += NSl*Nx*Ny;
            ind_mat += NSl*Nx*Ny;
        }
        ind_vec_out += NSl*Nx*Ny;
    }
}

__kernel void squarematvecmult_conj(__global float2* outvec,
                          __global float2* mat,
                          __global float2* invec,
                          const int vecdim
                          )
{
    size_t Nx = get_global_size(2), Ny = get_global_size(1);
    size_t NSl = get_global_size(0);
    size_t x = get_global_id(2), y = get_global_id(1);
    size_t k = get_global_id(0);
    
    size_t ind_vec_out = k*Nx*Ny+Nx*y + x;
    size_t ind_vec_in = k*Nx*Ny+Nx*y + x;
    size_t ind_mat = k*Nx*Ny+Nx*y + x;
    
    for (int dim_x=0; dim_x<vecdim; dim_x++)
    {
        outvec[ind_vec_out] = 0.0f;
        ind_vec_in = k*Nx*Ny+Nx*y + x;
        ind_mat = ind_vec_out;
        for (int dim_y=0; dim_y<vecdim; dim_y++)
        {
            outvec[ind_vec_out] += cmult_conj(invec[ind_vec_in], mat[ind_mat]);
            ind_vec_in += NSl*Nx*Ny;
            ind_mat += vecdim*NSl*Nx*Ny;
        }
        ind_vec_out += NSl*Nx*Ny;
    }
}

__kernel void extrapolate_x(
                __global float2 *xn1_,
                __global float2 *xn1,
                __global float2 *xn,
                const float theta)
{
    size_t i = get_global_id(0);
    xn1_[i] = xn1[i] * (1 + theta) - theta * xn[i];
}

__kernel void extrapolate_v(
                __global float8 *vn1_,
                __global float8 *vn1,
                __global float8 *vn,
                const float theta)
{
    size_t i = get_global_id(0);
    vn1_[i] = vn1[i] * (1 + theta) - theta * vn[i];
}

__kernel void update_x(
                __global float2 *xn1,
                __global float2 *xn,
                __global float2 *Kay,
                const float tau,
                const float theta)
{
    size_t i = get_global_id(0);
    xn1[i] = xn[i] - tau * (1 + theta) * Kay[i];
}

__kernel void update_x_explicit(
                __global float2 *xn1,
                __global float2 *xn,
                __global float2 *Kay,
                __global float2 *divz,
                const float tau,
                const float theta)
{
    size_t i = get_global_id(0);
    xn1[i] = xn[i] - tau * (1 + theta) * (Kay[i] - divz[i]);
}

__kernel void update_y(
                __global float2 *yn1,
                __global float2 *yn,
                __global float2 *Kx,
                __global float2 *dx,
                const float sigma,
                const float prox)
{
    size_t i = get_global_id(0);
    // float prox = 1.0 / (1.0 + sigma * lambdainv);
    yn1[i] = prox * (yn[i] + sigma * (Kx[i] - dx[i]));
}

__kernel void update_z_tv(
        __global float8 *zn1,
        __global float8 *zn,
        __global float8 *gx,
        const float sigma,
        const int NUk)
{
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = k*Nx*Ny+Nx*y + x;

  float abs_val = 0.0f;

  for (int uk=0; uk<NUk; uk++)
  {
      zn1[i] = zn[i] + sigma * gx[i];

      abs_val = hypot(abs_val,hypot(
        hypot(
          zn1[i].s0,
          zn1[i].s1
          ),
        hypot(
          hypot(
            zn1[i].s2,
            zn1[i].s3
            ),
          hypot(
            zn1[i].s4,
            zn1[i].s5
            )
          )
        ));
      i += NSl*Nx*Ny;
  }
  i = k*Nx*Ny+Nx*y + x;
  for (int uk=0; uk<NUk; uk++)
  {
      if (abs_val > 1.0f) zn1[i] /=abs_val;
      i += NSl*Nx*Ny;
  }
}

__kernel void update_v_explicit(
                __global float8 *v_new,
                __global float8 *v,
                __global float8 *z1,
                __global float8 *ez2,
                const float tau,
                const float theta)
{
    size_t i = get_global_id(0);
    v_new[i] = v[i] - tau * (1 + theta) * (ez2[i] - z1[i]);
}

__kernel void update_z1_tgv(
                __global float8 *z_new,
                __global float8 *z,
                __global float8 *gx,
                __global float8 *v,
                const float sigma,
                const float alphainv,
                const int NUk)
{
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = k*Nx*Ny+Nx*y + x;

  float fac = 0.0f;

  for (int uk=0; uk<NUk; uk++)
  {
      z_new[i] = z[i] + sigma * (gx[i] - v[i]);

        fac = hypot(fac,
          hypot(
            hypot(
              z_new[i].s0,
              z_new[i].s1
              ),
            hypot(
              hypot(
                z_new[i].s2,
                z_new[i].s3
                ),
              hypot(
                z_new[i].s4,
                z_new[i].s5
                )
              )
            )
          );
      i += NSl*Nx*Ny;
  }
  fac *= alphainv;
  //printf("fac: %2.2f\n", fac);
  i = k*Nx*Ny+Nx*y + x;
  for (int uk=0; uk<NUk; uk++)
  {
      if (fac > 1.0f) z_new[i] /=fac;
      i += NSl*Nx*Ny;
  }
}

__kernel void update_z2_tgv(
                    __global float16 *z_new,
                    __global float16 *z,
                    __global float16 *symgv,
                    const float sigma,
                    const float alphainv,
                    const int NUk)
{
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = k*Nx*Ny+Nx*y + x;

  float fac = 0.0f;

  for (int uk=0; uk<NUk; uk++)
  {
      z_new[i] = z[i] + sigma * symgv[i];

        fac = hypot(fac,hypot(
            hypot(
              hypot(
                hypot(
                  z_new[i].s0,
                  z_new[i].s1
                  ),
              hypot(
                z_new[i].s2,
                z_new[i].s3
                )
              ),
            hypot(
              z_new[i].s4,
              z_new[i].s5
              )
            ),
          hypot(
            hypot(
              2.0f*hypot(
                z_new[i].s6,
                z_new[i].s7
                ),
              2.0f*hypot(
                z_new[i].s8,
                z_new[i].s9
                )
              ),
            2.0f*hypot(
              z_new[i].sa,
              z_new[i].sb
              )
            )
          )
        );
        i+=NSl*Nx*Ny;
  }
  fac *= alphainv;
  i = k*Nx*Ny+Nx*y + x;
  for (int uk=0; uk<NUk; uk++)
  {
      if (fac > 1.0f) z_new[i] /=fac;
      i += NSl*Nx*Ny;
  }
}

__kernel void operator_fwd_ssense(
                    __global float2 *out,
                    __global float2 *in,
                    __global float2 *coils,
                    const int NCo,
                    const int Nmaps)
{
  size_t X = get_global_size(2);
  size_t Y = get_global_size(1);
  size_t NSl = get_global_size(0);

  size_t x = get_global_id(2);
  size_t y = get_global_id(1);
  size_t k = get_global_id(0);

  float2 tmp_in = 0.0f;
  float2 tmp_coil = 0.0f;

  for (int coil=0; coil < NCo; coil++)
  {
    float2 f_sum = 0.0f;
    for (int map=0; map < Nmaps; map++)
    {
      tmp_in = in[map*NSl*X*Y + k*X*Y+ y*X + x];
      tmp_coil = (float2) coils[map*NCo*NSl*X*Y + coil*NSl*X*Y + k*X*Y + y*X + x];
      f_sum += (float2)(tmp_in.x * tmp_coil.x - tmp_in.y * tmp_coil.y,
                        tmp_in.x * tmp_coil.y + tmp_in.y * tmp_coil.x);
    }
    out[coil*NSl*X*Y+ k*X*Y + y*X + x] = f_sum;
  }
}

__kernel void operator_ad_ssense(
                    __global float2 *out,
                    __global float2 *in,
                    __global float2 *coils,
                    const int NCo,
                    const int Nmaps)
{
  size_t X = get_global_size(2);
  size_t Y = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2);
  size_t y = get_global_id(1);
  size_t k = get_global_id(0);

  float2 tmp_in = 0.0f;
  float2 conj_coils = 0.0f;

  for (int map=0; map < Nmaps; map++)
  {
    float2 f_sum = 0.0f;
    for (int coil=0; coil < NCo; coil++)
    {
      tmp_in = in[coil*NSl*X*Y + k*X*Y+ y*X + x];
      conj_coils = (float2) (coils[map*NCo*NSl*X*Y + coil*NSl*X*Y + k*X*Y + y*X + x].x,
                            -coils[map*NCo*NSl*X*Y + coil*NSl*X*Y + k*X*Y + y*X + x].y);

      f_sum += (float2)(tmp_in.x*conj_coils.x-tmp_in.y*conj_coils.y,
                        tmp_in.x*conj_coils.y+tmp_in.y*conj_coils.x);
    }
    out[map*NSl*X*Y + k*X*Y + y*X + x] = f_sum;
  }
}

__kernel void update_Kyk1_ssense(
                    __global float2 *out,
                    __global float2 *in,
                    __global float2 *coils,
                    __global float8 *p,
                    const int NCo,
                    const int Nmaps,
                    __global float* ratio,
                    const float dz)
{
  size_t X = get_global_size(2);
  size_t Y = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2);
  size_t y = get_global_id(1);
  size_t k = get_global_id(0);

  float2 tmp_in = 0.0f;
  float2 conj_coils = 0.0f;

  size_t i = k*X*Y+X*y + x;

  for (int map=0; map < Nmaps; map++)
  {
    float2 f_sum = (float2) 0.0f;
    for (int coil=0; coil < NCo; coil++)
    {
      tmp_in = in[coil*NSl*X*Y + k*X*Y+ y*X + x];
      conj_coils = (float2) (coils[map*NCo*NSl*X*Y + coil*NSl*X*Y + k*X*Y + y*X + x].x,
                            -coils[map*NCo*NSl*X*Y + coil*NSl*X*Y + k*X*Y + y*X + x].y);

      f_sum += (float2)(tmp_in.x*conj_coils.x-tmp_in.y*conj_coils.y,
                        tmp_in.x*conj_coils.y+tmp_in.y*conj_coils.x);
    }
    // divergence
    float8 val = p[i];
    if (x == X-1)
    {
      //real
      val.s0 = 0.0f;
      //imag
      val.s1 = 0.0f;
    }
    if (x > 0)
    {
        //real
        val.s0 -= p[i-1].s0;
        //imag
        val.s1 -= p[i-1].s1;
    }
    if (y == Y-1)
    {
        //real
        val.s2 = 0.0f;
        //imag
        val.s3 = 0.0f;
    }
    if (y > 0)
    {
        //real
        val.s2 -= p[i-X].s2;
        //imag
        val.s3 -= p[i-X].s3;
    }
    if (k == NSl-1)
    {
        //real
        val.s4 = 0.0f;
        //imag
        val.s5 = 0.0f;
    }
    if (k > 0)
    {
        //real
        val.s4 -= p[i-X*Y].s4;
        //imag
        val.s5 -= p[i-X*Y].s5;
    }

    val*=ratio[map];

    out[map*NSl*X*Y + k*X*Y + y*X + x] = f_sum - (val.s01+val.s23+val.s45/dz);
    i += NSl*X*Y;
  }
}

__kernel void update_v(
                __global float2 *v,
                __global float2 *v_,
                __global float2 *Kyk2,
                const float tau
                )
{
    size_t i = get_global_id(0);
    v[i] = v_[i]-tau*Kyk2[i];
}


__kernel void update_r(
                __global float2 *r,
                __global float2 *r_,
                __global float2 *A,
                __global float2 *A_,
                __global float2 *res,
                const float sigma,
                const float theta,
                const float lambdainv
                )
{
    size_t i = get_global_id(0);
    r[i] = (r_[i]+sigma*((1+theta)*A[i]-theta*A_[i] - res[i]))*lambdainv;
}


__kernel void update_z2(
                __global float16 *z_new,
                __global float16 *z,
                __global float16 *gx,
                __global float16 *gx_,
                const float sigma,
                const float theta,
                const float alphainv,
                const int NUk,
                __global float* ratio
                )
{
    size_t Nx = get_global_size(2), Ny = get_global_size(1);
    size_t NSl = get_global_size(0);
    size_t x = get_global_id(2), y = get_global_id(1);
    size_t k = get_global_id(0);
    size_t i = k*Nx*Ny+Nx*y + x;

    float fac = 0.0f;
    float16 square = 0.0f;

    for (int uk=0; uk<NUk; uk++)
    {
//         fac = 0.0f;
        z_new[i] = z[i] + sigma*((1+theta)*gx[i]-theta*gx_[i]);

        // reproject
//         square = powr(z_new[i], 2);
//         fac += (square.s0+square.s1+square.s2+square.s3+square.s4+square.s5
//                     +4.0f*(square.s6+square.s7+square.s8+square.s9+square.sa+square.sb));
        fac = hypot(fac,hypot(
            hypot(
              hypot(
                hypot(
                  z_new[i].s0,
                  z_new[i].s1
                  ),
              hypot(
                z_new[i].s2,
                z_new[i].s3
                )
              ),
            hypot(
              z_new[i].s4,
              z_new[i].s5
              )
            ),
          hypot(
            hypot(
              2.0f*hypot(
                z_new[i].s6,
                z_new[i].s7
                ),
              2.0f*hypot(
                z_new[i].s8,
                z_new[i].s9
                )
              ),
            2.0f*hypot(
              z_new[i].sa,
              z_new[i].sb
              )
            )
          )
        );

        i += NSl*Nx*Ny;
    }
//     fac = sqrt(fac);
    fac *= alphainv;
    i = k*Nx*Ny+Nx*y + x;
    for (int uk=0; uk<NUk; uk++)
    {
        if (fac > 1.0f) z_new[i] /= fac;
        i += NSl*Nx*Ny;
    }
}


__kernel void update_z1(
                __global float8 *z_new,
                __global float8 *z,
                __global float8 *gx,
                __global float8 *gx_,
                __global float8 *vx,
                __global float8 *vx_,
                const float sigma,
                const float theta,
                const float alphainv,
                const int NUk_tgv,
                const int NUk_H1,
                const float h1inv,
                __global float* ratio
                )
{
    size_t Nx = get_global_size(2), Ny = get_global_size(1);
    size_t NSl = get_global_size(0);
    size_t x = get_global_id(2), y = get_global_id(1);
    size_t k = get_global_id(0);
    size_t i = k*Nx*Ny+Nx*y + x;

    float fac = 0.0f;
    float8 square = 0.0f;

    for (int uk=0; uk<NUk_tgv; uk++)
    {
//         fac = 0.0f;
        z_new[i] = z[i] + sigma*(
            (1+theta)*gx[i]-theta*gx_[i]-((1+theta)*vx[i]-theta*vx_[i]));

        // reproject
//         square = powr(z_new[i], 2);
//         fac += (square.s0+square.s1+square.s2+square.s3+square.s4+square.s5);
        fac = hypot(fac,
          hypot(
            hypot(
              z_new[i].s0,
              z_new[i].s1
              ),
            hypot(
              hypot(
                z_new[i].s2,
                z_new[i].s3
                ),
              hypot(
                z_new[i].s4,
                z_new[i].s5
                )
              )
            )
          );
        i += NSl*Nx*Ny;
    }
//     fac = sqrt(fac);
    fac *= alphainv;
    i = k*Nx*Ny+Nx*y + x;
    for (int uk=0; uk<NUk_tgv; uk++)
    {
        if (fac > 1.0f) z_new[i] /= fac;
        i += NSl*Nx*Ny;
    }
    i = NSl*Nx*Ny*NUk_tgv+k*Nx*Ny+Nx*y + x;
    for (int uk=NUk_tgv; uk<(NUk_tgv+NUk_H1); uk++)
    {
        z_new[i] = (z[i] + sigma*((1+theta)*gx[i]-theta*gx_[i]))*h1inv;
        i += NSl*Nx*Ny;
    }
}

__kernel void update_z1_tv(
                __global float8 *z_new,
                __global float8 *z,
                __global float8 *gx,
                __global float8 *gx_,
                const float sigma, const float theta, const float alphainv,
                const int NUk_tgv, const int NUk_H1, const float h1inv,
                __global float* ratio
                )
{
    size_t Nx = get_global_size(2), Ny = get_global_size(1);
    size_t NSl = get_global_size(0);
    size_t x = get_global_id(2), y = get_global_id(1);
    size_t k = get_global_id(0);
    size_t i = k*Nx*Ny+Nx*y + x;

    float fac = 0.0f;
    float8 square = 0.0f;

    for (int uk=0; uk<NUk_tgv; uk++)
    {
//         fac = 0.0f;
        z_new[i] = z[i] + sigma*((1+theta)*gx[i]-theta*gx_[i]);

        // reproject
//         square = powr(z_new[i], 2);
//         fac += sqrt(square.s0+square.s1+square.s2+square.s3+square.s4+square.s5);///ratio[uk];
        fac = hypot(fac,
          hypot(
            hypot(
              z_new[i].s0,
              z_new[i].s1
              ),
            hypot(
              hypot(
                z_new[i].s2,
                z_new[i].s3
                ),
              hypot(
                z_new[i].s4,
                z_new[i].s5
                )
              )
            )
          );

        i += NSl*Nx*Ny;
    }
    fac *= alphainv;
    i = k*Nx*Ny+Nx*y + x;
    for (int uk=0; uk<NUk_tgv; uk++)
    {
        if (fac > 1.0f){z_new[i] /= fac;}
        i += NSl*Nx*Ny;
    }
    i = NSl*Nx*Ny*NUk_tgv+k*Nx*Ny+Nx*y + x;
    for (int uk=NUk_tgv; uk<(NUk_tgv+NUk_H1); uk++)
    {
        z_new[i] = (z[i] + sigma*((1+theta)*gx[i]-theta*gx_[i]))*h1inv;
        i += NSl*Nx*Ny;
    }
}


__kernel void update_primal_precond(
__global float2 *u_new,
__global float2 *u,
__global float2 *Kyk,
__global float2 *u_k,
const float tau,
const float tauinv,
__global float* min,
__global float* max,
__global int* real, const int NUk
)
{
    size_t Nx = get_global_size(2), Ny = get_global_size(1);
    size_t NSl = get_global_size(0);
    size_t x = get_global_id(2), y = get_global_id(1);
    size_t k = get_global_id(0);
    size_t i = k*Nx*Ny+Nx*y + x;
    float norm = 0;

    for (int uk=0; uk<NUk; uk++)
    {
        u_new[i] = (u[i]-tau*Kyk[i]+tauinv*u_k[i])/(1+tauinv);
        i += NSl*Nx*Ny;
    }
}

__kernel void update_primal(
__global float2 *u_new,
__global float2 *u,
__global float2 *Kyk,
__global float2 *u_k,
const float tau,
const float tauinv,
__global float* min,
__global float* max,
__global int* real, const int NUk
)
{
    size_t Nx = get_global_size(2), Ny = get_global_size(1);
    size_t NSl = get_global_size(0);
    size_t x = get_global_id(2), y = get_global_id(1);
    size_t k = get_global_id(0);
    size_t i = k*Nx*Ny+Nx*y + x;
    float norm = 0;

    for (int uk=0; uk<NUk; uk++)
    {
        u_new[i] = (u[i]-tau*Kyk[i]+tauinv*u_k[i])/(1+tauinv);

        if(real[uk]>=1)
        {
            u_new[i].s1 = 0.0f;
            if (u_new[i].s0<min[uk])
            {
                u_new[i].s0 = min[uk];
            }
            if(u_new[i].s0>max[uk])
            {
                u_new[i].s0 = max[uk];
            }
        }
        else
        {
            norm =  sqrt(
              pow(
                (float)(u_new[i].s0),(float)(2.0))
              + pow((float)(u_new[i].s1),(float)(2.0)));
            if (norm<min[uk])
            {
                u_new[i].s0 *= 1/norm*min[uk];
                u_new[i].s1 *= 1/norm*min[uk];
            }
            if(norm>max[uk])
            {
                u_new[i].s0 *= 1/norm*max[uk];
                u_new[i].s1 *= 1/norm*max[uk];
            }
        }
        i += NSl*Nx*Ny;
    }
}

__kernel void update_box(
__global float2 *u,
__global float* min,
__global float* max,
__global int* real, const int NUk
)
{
    size_t Nx = get_global_size(2), Ny = get_global_size(1);
    size_t NSl = get_global_size(0);
    size_t x = get_global_id(2), y = get_global_id(1);
    size_t k = get_global_id(0);
    size_t i = k*Nx*Ny+Nx*y + x;
    float norm = 0;

    for (int uk=0; uk<NUk; uk++)
    {
        if(real[uk]>=1)
        {
            u[i].s1 = 0.0f;
            if (u[i].s0<min[uk])
            {
                u[i].s0 = min[uk];
            }
            if(u[i].s0>max[uk])
            {
                u[i].s0 = max[uk];
            }
        }
        else
        {
            norm =  sqrt(
              pow(
                (float)(u[i].s0),(float)(2.0))
              + pow((float)(u[i].s1),(float)(2.0)));
            if (norm<min[uk])
            {
                u[i].s0 *= 1/norm*min[uk];
                u[i].s1 *= 1/norm*min[uk];
            }
            if(norm>max[uk])
            {
                u[i].s0 *= 1/norm*max[uk];
                u[i].s1 *= 1/norm*max[uk];
            }
        }
        i += NSl*Nx*Ny;
    }
}


__kernel void update_primal_LM(
                __global float2 *u_new,
                __global float2 *u,
                __global float2 *Kyk,
                __global float2 *u_k,
                __global float* A,
                const float tau,
                const float tauinv,
                __global float* min,
                __global float* max,
                __global int* real,
                const int NUk,
                __global float* ratio
                )
{
    size_t Nx = get_global_size(2), Ny = get_global_size(1);
    size_t NSl = get_global_size(0);
    size_t x = get_global_id(2), y = get_global_id(1);
    size_t k = get_global_id(0);
    size_t i = k*Nx*Ny+Nx*y + x;
    float norm = 0;

    for (int uk=0; uk<NUk; uk++)
    {
        u_new[i] = (u[i]-tau*Kyk[i]+tauinv*A[i]*u_k[i])/(1+tauinv*A[i]);

        i += NSl*Nx*Ny;
    }
}


__kernel void gradient(
                __global float8 *grad,
                __global float2 *u,
                const int NUk,
                const float dz,
                __global float* ratio
                )
{
    size_t Nx = get_global_size(2), Ny = get_global_size(1);
    size_t NSl = get_global_size(0);
    size_t x = get_global_id(2), y = get_global_id(1);
    size_t k = get_global_id(0);
    size_t i = k*Nx*Ny+Nx*y + x;
    
    float2 tmp_u = 0.0f;
    
    for (int uk=0; uk<NUk; uk++)
    {
        // gradient
        grad[i] = (float8)(-u[i],-u[i],-u[i]*dz,0.0f,0.0f);
        if (x < Nx-1)
        {
            grad[i].s01 += u[i+1];
        }
        else
        {
            grad[i].s01 = 0.0f;
        }

        if (y < Ny-1)
        {
            grad[i].s23 += u[i+Nx];
        }
        else
        {
            grad[i].s23 = 0.0f;
        }
        if (k < NSl-1)
        {
            grad[i].s45 += u[i+Nx*Ny]*dz;
                }
        else
        {
            grad[i].s45 = 0.0f;
        }
        grad[i] *= ratio[uk];
        i += NSl*Nx*Ny;
    }
}


__kernel void sym_grad(
                __global float16 *sym,
                __global float8 *w,
                const int NUk,
                const float dz
                )
{
    size_t Nx = get_global_size(2), Ny = get_global_size(1);
    size_t NSl = get_global_size(0);
    size_t x = get_global_id(2), y = get_global_id(1);
    size_t k = get_global_id(0);
    size_t i = k*Nx*Ny+Nx*y + x;


    for (int uk=0; uk<NUk; uk++)
    {
        // symmetrized gradient
        float16 val_real = (float16)(
            w[i].s024, w[i].s024, w[i].s024,
            0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f);
        float16 val_imag = (float16)(
            w[i].s135, w[i].s135, w[i].s135,
            0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f);
        if (x > 0)
        {
            val_real.s012 -= w[i-1].s024;
            val_imag.s012 -= w[i-1].s135;
        }
        else
        {
            val_real.s012 = (float3) 0.0f;
            val_imag.s012 = (float3) 0.0f;
        }

        if (y > 0)
        {
            val_real.s345 -= w[i-Nx].s024;
            val_imag.s345 -= w[i-Nx].s135;
        }
        else
        {
            val_real.s345 = (float3) 0.0f;
            val_imag.s345 = (float3) 0.0f;
        }

        if (k > 0)
        {
            val_real.s678 -= w[i-Nx*Ny].s024;
            val_imag.s678 -= w[i-Nx*Ny].s135;
        }
        else
        {
            val_real.s678 = (float3) 0.0f;
            val_imag.s678 = (float3) 0.0f;
        }

        sym[i] = (float16)(
          val_real.s0, val_imag.s0,
          val_real.s4, val_imag.s4,
          val_real.s8*dz, val_imag.s8*dz,
          0.5f*(val_real.s1 + val_real.s3),
          0.5f*(val_imag.s1 + val_imag.s3),
          0.5f*(val_real.s2 + val_real.s6*dz),
          0.5f*(val_imag.s2 + val_imag.s6*dz),
          0.5f*(val_real.s5 + val_real.s7*dz),
          0.5f*(val_imag.s5 + val_imag.s7*dz),
          0.0f,0.0f,0.0f,0.0f);
        i += NSl*Nx*Ny;
    }
}


__kernel void divergence(
                __global float2 *div,
                __global float8 *p,
                const int NUk,
                const float dz,
                __global float* ratio
                )
{
    size_t Nx = get_global_size(2), Ny = get_global_size(1);
    size_t NSl = get_global_size(0);
    size_t x = get_global_id(2), y = get_global_id(1);
    size_t k = get_global_id(0);
    size_t i = k*Nx*Ny+Nx*y + x;
    

    for (int uk=0; uk<NUk; uk++)
    {
        // divergence
        float8 val = p[i];
        if (x == Nx-1)
        {
            //real
            val.s0 = 0.0f;
            //imag
            val.s1 = 0.0f;
        }
        if (x > 0)
        {
            //real
            val.s01 -= p[i-1].s01;
            //imag
//             val.s1 -= p[j-1].s1*ratio[j-1];
        }
        if (y == Ny-1)
        {
            //real
            val.s2 = 0.0f;
            //imag
            val.s3 = 0.0f;
        }
        if (y > 0)
        {
            //real
            val.s23 -= p[i-Nx].s23;
            //imag
//             val.s3 -= p[j-Nx].s3*ratio[j-Nx];
        }
        if (k == NSl-1)
        {
            //real
            val.s4 = 0.0f;
            //imag
            val.s5 = 0.0f;
        }
        if (k > 0)
        {
            //real
            val.s45 -= p[i-Nx*Ny].s45;
            //imag
//             val.s5 -= p[j-Nx*Ny].s5*ratio[j-Nx*Ny];
        }
        div[i] = (val.s01+val.s23+val.s45*dz)*ratio[uk];
        // scale gradients
        i += NSl*Nx*Ny;
    }
}


__kernel void sym_divergence(
                __global float8 *w,
                __global float16 *q,
                const int NUk,
                const float dz
                )
{
    size_t Nx = get_global_size(2), Ny = get_global_size(1);
    size_t NSl = get_global_size(0);
    size_t x = get_global_id(2), y = get_global_id(1);
    size_t k = get_global_id(0);
    size_t i = k*Nx*Ny+Nx*y + x;

    for (int uk=0; uk<NUk; uk++)
    {
        // divergence
        float16 val0 = -q[i];
        float16 val_real = (float16)(
            val0.s0, val0.s6, val0.s8,
            val0.s6, val0.s2, val0.sa,
            val0.s8, val0.sa, val0.s4,
            0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f);
        float16 val_imag = (float16)(
            val0.s1, val0.s7, val0.s9,
            val0.s7, val0.s3, val0.sb,
            val0.s9, val0.sb, val0.s5,
            0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f);
        if (x == 0)
        {
            //real
            val_real.s012 = 0.0f;
            //imag
            val_imag.s012 = 0.0f;
        }
        if (x < Nx-1)
        {
            //real
            val_real.s012 += (float3)(q[i+1].s0, q[i+1].s68);
            //imag
            val_imag.s012 += (float3)(q[i+1].s1, q[i+1].s79);
        }
        if (y == 0)
        {
            //real
            val_real.s345 = 0.0f;
            //imag
            val_imag.s345 = 0.0f;
        }
        if (y < Ny-1)
        {
            //real
            val_real.s345 += (float3)(q[i+Nx].s6, q[i+Nx].s2, q[i+Nx].sa);
            //imag
            val_imag.s345 += (float3)(q[i+Nx].s7, q[i+Nx].s3, q[i+Nx].sb);
        }
        if (k == 0)
        {
        //real
            val_real.s678 = 0.0f;
            //imag
            val_imag.s678 = 0.0f;
        }
        if (k < NSl-1)
        {
            //real
            val_real.s678 += (float3)(q[i+Nx*Ny].s8a, q[i+Nx*Ny].s4);
            //imag
            val_imag.s678 += (float3)(q[i+Nx*Ny].s9b, q[i+Nx*Ny].s5);
        }
        // linear step
        //real
        w[i].s024 = val_real.s012 + val_real.s345 + val_real.s678*dz;
        //imag
        w[i].s135 = val_imag.s012 + val_imag.s345 + val_imag.s678*dz;

        i += NSl*Nx*Ny;
    }
}


__kernel void update_Kyk2(
                __global float8 *w,
                __global float16 *q,
                __global float8 *z,
                const int NUk,
                const int first,
                const float dz
                )
{
    size_t Nx = get_global_size(2), Ny = get_global_size(1);
    size_t NSl = get_global_size(0);
    size_t x = get_global_id(2), y = get_global_id(1);
    size_t k = get_global_id(0);
    size_t i = k*Nx*Ny+Nx*y + x;

    for (int uk=0; uk<NUk; uk++)
    {
        // divergence
        float16 val0 = -q[i];
        float16 val_real = (float16)(
                    val0.s0, val0.s6, val0.s8,
                    val0.s6, val0.s2, val0.sa,
                    val0.s8, val0.sa, val0.s4,
                    0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f);
        float16 val_imag = (float16)(
                    val0.s1, val0.s7, val0.s9,
                    val0.s7, val0.s3, val0.sb,
                    val0.s9, val0.sb, val0.s5,
                    0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f);
        if (x == 0)
        {
            //real
            val_real.s012 = 0.0f;
            //imag
            val_imag.s012 = 0.0f;
        }
        if (x < Nx-1)
        {
            //real
            val_real.s012 += (float3)(q[i+1].s0, q[i+1].s68);
            //imag
            val_imag.s012 += (float3)(q[i+1].s1, q[i+1].s79);
        }
        if (y == 0)
        {
            //real
            val_real.s345 = 0.0f;
            //imag
            val_imag.s345 = 0.0f;
        }
        if (y < Ny-1)
        {
            //real
            val_real.s345 += (float3)(q[i+Nx].s6, q[i+Nx].s2, q[i+Nx].sa);
            //imag
            val_imag.s345 += (float3)(q[i+Nx].s7, q[i+Nx].s3, q[i+Nx].sb);
        }
        if (k == 0)
        {
            //real
            val_real.s678 = 0.0f;
            //imag
            val_imag.s678 = 0.0f;
        }
        if (k < NSl-1)
        {
            //real
            val_real.s678 += (float3)(q[i+Nx*Ny].s8a, q[i+Nx*Ny].s4);
            //imag
            val_imag.s678 += (float3)(q[i+Nx*Ny].s9b, q[i+Nx*Ny].s5);
        }
        // linear step

        //real
        w[i].s024 = - val_real.s012
                    - val_real.s345
                    - val_real.s678*dz
                    -z[i].s024;
        //imag
        w[i].s135 = - val_imag.s012
                    - val_imag.s345
                    - val_imag.s678*dz
                    -z[i].s135;
                    
        i += NSl*Nx*Ny;
    }
}


__kernel void operator_fwd(
                __global float2 *out,
                __global float2 *in,
                __global float2 *coils,
                __global float2 *grad,
                const int NCo,
                const int NScan,
                const int Nuk
                )
{
    size_t X = get_global_size(2);
    size_t Y = get_global_size(1);
    size_t NSl = get_global_size(0);

    size_t x = get_global_id(2);
    size_t y = get_global_id(1);
    size_t k = get_global_id(0);

    float2 tmp_in = 0.0f;
    float2 tmp_grad = 0.0f;
    float2 tmp_coil = 0.0f;
    float2 tmp_mul = 0.0f;


    for (int scan=0; scan<NScan; scan++)
    {
        for (int coil=0; coil < NCo; coil++)
        {
            tmp_coil = coils[coil*NSl*X*Y + k*X*Y + y*X + x];
            float2 sum = 0.0f;
            for (int uk=0; uk<Nuk; uk++)
            {
                tmp_grad = grad[
                    uk*NScan*NSl*X*Y+scan*NSl*X*Y + k*X*Y + y*X + x];
                tmp_in = in[uk*NSl*X*Y+k*X*Y+ y*X + x];

                tmp_mul = (float2)(
                        tmp_in.x*tmp_grad.x-tmp_in.y*tmp_grad.y,
                        tmp_in.x*tmp_grad.y+tmp_in.y*tmp_grad.x);
                sum += (float2)(
                        tmp_mul.x*tmp_coil.x-tmp_mul.y*tmp_coil.y,
                        tmp_mul.x*tmp_coil.y+tmp_mul.y*tmp_coil.x);

            }
            out[scan*NCo*NSl*X*Y+coil*NSl*X*Y+k*X*Y + y*X + x] = sum;
        }
    }
}


__kernel void operator_ad(
                __global float2 *out,
                __global float2 *in,
                __global float2 *coils,
                __global float2 *grad,
                const int NCo,
                const int NScan,
                const int Nuk
                )
{
    size_t X = get_global_size(2);
    size_t Y = get_global_size(1);
    size_t NSl = get_global_size(0);

    size_t x = get_global_id(2);
    size_t y = get_global_id(1);
    size_t k = get_global_id(0);


    float2 tmp_in = 0.0f;
    float2 tmp_mul = 0.0f;
    float2 conj_grad = 0.0f;
    float2 conj_coils = 0.0f;
    
    for (int uk=0; uk<Nuk; uk++)
    {
        float2 sum = (float2) 0.0f;
        for (int scan=0; scan<NScan; scan++)
        {
            conj_grad = (float2)(
                    grad[uk*NScan*NSl*X*Y+scan*NSl*X*Y + k*X*Y + y*X + x].x,
                    -grad[uk*NScan*NSl*X*Y+scan*NSl*X*Y + k*X*Y + y*X + x].y);
            for (int coil=0; coil < NCo; coil++)
            {
                conj_coils = (float2)(
                        coils[coil*NSl*X*Y + k*X*Y + y*X + x].x,
                        -coils[coil*NSl*X*Y + k*X*Y + y*X + x].y);

                tmp_in = in[scan*NCo*NSl*X*Y+coil*NSl*X*Y + k*X*Y+ y*X + x];
                tmp_mul = (float2)(
                        tmp_in.x*conj_grad.x-tmp_in.y*conj_grad.y,
                        tmp_in.x*conj_grad.y+tmp_in.y*conj_grad.x);


                sum += (float2)(
                        tmp_mul.x*conj_coils.x-tmp_mul.y*conj_coils.y,
                        tmp_mul.x*conj_coils.y+tmp_mul.y*conj_coils.x);
            }
        }
        out[uk*NSl*X*Y+k*X*Y+y*X+x] = sum;
    }
}


__kernel void update_Kyk1(
                __global float2 *out,
                __global float2 *in,
                __global float2 *coils,
                __global float2 *grad,
                __global float8 *p,
                const int NCo,
                const int NScan,
                const int NUk,
                const float dz,
                __global float* ratio
                )
{
    size_t X = get_global_size(2);
    size_t Y = get_global_size(1);
    size_t NSl = get_global_size(0);

    size_t x = get_global_id(2);
    size_t y = get_global_id(1);
    size_t k = get_global_id(0);

    size_t i = k*X*Y+X*y + x;

    float2 tmp_in = 0.0f;
    float2 tmp_mul = 0.0f;
    float2 conj_grad = 0.0f;
    float2 conj_coils = 0.0f;
    


    for (int uk=0; uk<NUk; uk++)
    {
        float2 sum = (float2) 0.0f;
        for (int scan=0; scan<NScan; scan++)
        {
            conj_grad = (float2)(
                grad[uk*NScan*NSl*X*Y+scan*NSl*X*Y + k*X*Y + y*X + x].x,
                -grad[uk*NScan*NSl*X*Y+scan*NSl*X*Y + k*X*Y + y*X + x].y);
            for (int coil=0; coil < NCo; coil++)
            {
                conj_coils = (float2)(
                    coils[coil*NSl*X*Y + k*X*Y + y*X + x].x,
                    -coils[coil*NSl*X*Y + k*X*Y + y*X + x].y);

                tmp_in = in[scan*NCo*NSl*X*Y+coil*NSl*X*Y + k*X*Y+ y*X + x];
                tmp_mul = (float2)(
                    tmp_in.x*conj_grad.x-tmp_in.y*conj_grad.y,
                    tmp_in.x*conj_grad.y+tmp_in.y*conj_grad.x);

                sum += (float2)(
                    tmp_mul.x*conj_coils.x-tmp_mul.y*conj_coils.y,
                    tmp_mul.x*conj_coils.y+tmp_mul.y*conj_coils.x);
            }
        }
        // divergence
        float8 val = p[i];
        if (x == X-1)
        {
            val.s01 = 0.0f;
        }
        if (x > 0)
        {
            val.s01 -= p[i-1].s01;
        }
        if (y == Y-1)
        {
            val.s23 = 0.0f;
        }
        if (y > 0)
        {
            val.s23 -= p[i-X].s23;
        }
        if (k == NSl-1)
        {
            val.s45 = 0.0f;
        }
        if (k > 0)
        {
            val.s45 -= p[i-X*Y].s45;
        }

        out[uk*NSl*X*Y+k*X*Y+y*X+x] = sum - (val.s01+val.s23+val.s45*dz)*ratio[uk];
        i += NSl*X*Y;
    }
}


__kernel void operator_fwd_imagespace(
                __global float2 *out,
                __global float2 *in,
                __global float2 *grad,
                const int NScan,
                const int Nuk
                )
{
    size_t X = get_global_size(2);
    size_t Y = get_global_size(1);
    size_t NSl = get_global_size(0);

    size_t x = get_global_id(2);
    size_t y = get_global_id(1);
    size_t k = get_global_id(0);

    float2 tmp_in = 0.0f;
    float2 tmp_grad = 0.0f;

    for (int scan=0; scan<NScan; scan++)
    {
        float2 sum = 0.0f;
        for (int uk=0; uk<Nuk; uk++)
        {
            tmp_grad = grad[uk*NScan*NSl*X*Y+scan*NSl*X*Y + k*X*Y + y*X + x];
            tmp_in = in[uk*NSl*X*Y+k*X*Y+ y*X + x];

            sum += (float2)(
                tmp_in.x*tmp_grad.x-tmp_in.y*tmp_grad.y,
                tmp_in.x*tmp_grad.y+tmp_in.y*tmp_grad.x);

        }
        out[scan*NSl*X*Y+k*X*Y + y*X + x] = sum;
    }
}


__kernel void operator_ad_imagespace(
                __global float2 *out,
                __global float2 *in,
                __global float2 *grad,
                const int NScan,
                const int Nuk
                )
{
    size_t X = get_global_size(2);
    size_t Y = get_global_size(1);
    size_t NSl = get_global_size(0);

    size_t x = get_global_id(2);
    size_t y = get_global_id(1);
    size_t k = get_global_id(0);


    float2 tmp_in = 0.0f;
    float2 conj_grad = 0.0f;



    for (int uk=0; uk<Nuk; uk++)
    {
        float2 sum = (float2) 0.0f;
        for (int scan=0; scan<NScan; scan++)
        {
            conj_grad = (float2)(
                    grad[uk*NScan*NSl*X*Y+scan*NSl*X*Y + k*X*Y + y*X + x].x,
                    -grad[uk*NScan*NSl*X*Y+scan*NSl*X*Y + k*X*Y + y*X + x].y);
            tmp_in = in[scan*NSl*X*Y+ k*X*Y+ y*X + x];
            sum += (float2)(
                    tmp_in.x*conj_grad.x-tmp_in.y*conj_grad.y,
                    tmp_in.x*conj_grad.y+tmp_in.y*conj_grad.x);

        }
        out[uk*NSl*X*Y+k*X*Y+y*X+x] = sum;
    }
}



__kernel void update_Kyk1_imagespace(
                __global float2 *out,
                __global float2 *in,
                __global float2 *grad,
                __global float8 *p,
                const int NScan,
                const int Nuk,
                const float dz,
                __global float* ratio
                )
{
    size_t X = get_global_size(2);
    size_t Y = get_global_size(1);
    size_t NSl = get_global_size(0);

    size_t x = get_global_id(2);
    size_t y = get_global_id(1);
    size_t k = get_global_id(0);

    size_t i = k*X*Y+X*y + x;

    float2 tmp_in = 0.0f;
    float2 conj_grad = 0.0f;

    for (int uk=0; uk<Nuk; uk++)
    {
        float2 sum = (float2) 0.0f;
        for (int scan=0; scan<NScan; scan++)
        {
          conj_grad = (float2)(
                  grad[uk*NScan*NSl*X*Y+scan*NSl*X*Y + k*X*Y + y*X + x].x,
                  -grad[uk*NScan*NSl*X*Y+scan*NSl*X*Y + k*X*Y + y*X + x].y);
          tmp_in = in[scan*NSl*X*Y+ k*X*Y+ y*X + x];
          sum += (float2)(
                  tmp_in.x*conj_grad.x-tmp_in.y*conj_grad.y,
                  tmp_in.x*conj_grad.y+tmp_in.y*conj_grad.x);
        }
        // divergence
        float8 val = p[i];
        if (x == X-1)
        {
            //real
            val.s0 = 0.0f;
            //imag
            val.s1 = 0.0f;
        }
        if (x > 0)
        {
            //real
            val.s0 -= p[i-1].s0;
            //imag
            val.s1 -= p[i-1].s1;
        }
        if (y == Y-1)
        {
            //real
            val.s2 = 0.0f;
            //imag
            val.s3 = 0.0f;
        }
        if (y > 0)
        {
            //real
            val.s2 -= p[i-X].s2;
            //imag
            val.s3 -= p[i-X].s3;
        }
        if (k == NSl-1)
        {
            //real
            val.s4 = 0.0f;
            //imag
            val.s5 = 0.0f;
        }
        if (k > 0)
        {
            //real
            val.s4 -= p[i-X*Y].s4;
            //imag
            val.s5 -= p[i-X*Y].s5;
        }
        // scale gradients
        val*=ratio[uk];
        out[uk*NSl*X*Y+k*X*Y+y*X+x] = sum - (val.s01+val.s23+val.s45*dz);
        i += NSl*X*Y;
    }
}


__kernel void update_primal_explicit(
                __global float2 *u_new,
                __global float2 *u,
                __global float2 *Kyk,
                __global float2 *u_k,
                __global float2* ATd,
                const float tau,
                const float delta_inv,
                const float lambd,
                __global float* mmin,
                __global float* mmax,
                __global int* real,
                const int NUk
                )
{
    size_t Nx = get_global_size(2), Ny = get_global_size(1);
    size_t NSl = get_global_size(0);
    size_t x = get_global_id(2), y = get_global_id(1);
    size_t k = get_global_id(0);
    size_t i = k*Nx*Ny+Nx*y + x;


    for (int uk=0; uk<NUk; uk++)
    {
        u_new[i] = u[i]-tau*(
          lambd*u_new[i]-lambd*ATd[i]+delta_inv*u[i]-delta_inv*u_k[i]-Kyk[i]);

        if(real[uk]>0)
        {
            u_new[i].s1 = 0;
            if (u_new[i].s0<mmin[uk])
            {
                u_new[i].s0 = mmin[uk];
            }
            if(u_new[i].s0>mmax[uk])
            {
                u_new[i].s0 = mmax[uk];
            }
        }
        else
        {
            if (u_new[i].s0<mmin[uk])
            {
                u_new[i].s0 = mmin[uk];
            }
            if(u_new[i].s0>mmax[uk])
            {
                u_new[i].s0 = mmax[uk];
            }
            if (u_new[i].s1<mmin[uk])
            {
                u_new[i].s1 = mmin[uk];
            }
            if(u_new[i].s1>mmax[uk])
            {
                u_new[i].s1 = mmax[uk];
            }
        }
        i += NSl*Nx*Ny;
    }
}


__kernel void operator_fwd_cg(
                __global float2 *out,
                __global float2 *in,
                __global float2 *coils,
                const int NCo,
                const int NScan
                )
{
    size_t X = get_global_size(2);
    size_t Y = get_global_size(1);
    size_t NSl = get_global_size(0);

    size_t x = get_global_id(2);
    size_t y = get_global_id(1);
    size_t k = get_global_id(0);

    float2 tmp_in = 0.0f;
    float2 tmp_grad = 0.0f;
    float2 tmp_coil = 0.0f;
    float2 tmp_mul = 0.0f;


    for (int scan=0; scan<NScan; scan++)
    {
        for (int coil=0; coil < NCo; coil++)
        {
              out[scan*NCo*NSl*X*Y+coil*NSl*X*Y+k*X*Y + y*X + x] = (float2)(
                      in[scan*NSl*X*Y+k*X*Y+ y*X + x].x
                      * coils[coil*NSl*X*Y + k*X*Y + y*X + x].x
                      - in[scan*NSl*X*Y+k*X*Y+ y*X + x].y
                      * coils[coil*NSl*X*Y + k*X*Y + y*X + x].y,
                      in[scan*NSl*X*Y+k*X*Y+ y*X + x].x
                      * coils[coil*NSl*X*Y + k*X*Y + y*X + x].y
                      + in[scan*NSl*X*Y+k*X*Y+ y*X + x].y
                      * coils[coil*NSl*X*Y + k*X*Y + y*X + x].x);
        }
    }
}


__kernel void operator_ad_cg(
                __global float2 *out,
                __global float2 *in,
                __global float2 *coils,
                const int NCo,
                const int NScan
                )
{
    size_t X = get_global_size(2);
    size_t Y = get_global_size(1);
    size_t NSl = get_global_size(0);
    size_t x = get_global_id(2);
    size_t y = get_global_id(1);
    size_t k = get_global_id(0);


    float2 tmp_in = 0.0f;
    float2 tmp_mul = 0.0f;
    float2 conj_grad = 0.0f;
    float2 conj_coils = 0.0f;


    for (int scan=0; scan<NScan; scan++)
    {
        float2 sum = (float2) 0.0f;
        for (int coil=0; coil < NCo; coil++)
        {
            conj_coils = (float2)(
                            coils[coil*NSl*X*Y + k*X*Y + y*X + x].x,
                            -coils[coil*NSl*X*Y + k*X*Y + y*X + x].y);
            tmp_in = in[scan*NCo*NSl*X*Y+coil*NSl*X*Y + k*X*Y+ y*X + x];

            sum += (float2)(
                    tmp_in.x*conj_coils.x-tmp_in.y*conj_coils.y,
                    tmp_in.x*conj_coils.y+tmp_in.y*conj_coils.x);
        }
        out[scan*NSl*X*Y+k*X*Y+y*X+x] = sum;
    }
}



__kernel void gradient_w_time(
                __global float8 *grad,
                __global float2 *u,
                const int NUk,
                const float dz,
                const float mu_1,
                __global float* dt
                )
{
    size_t Nx = get_global_size(2), Ny = get_global_size(1);
    size_t NSl = get_global_size(0);
    size_t x = get_global_id(2), y = get_global_id(1);
    size_t k = get_global_id(0);
    size_t i = k*Nx*Ny+Nx*y + x;
    
    float2 tmp_u = 0.0f;
    
    for (int uk=0; uk<NUk; uk++)
    {
        // gradient
        grad[i] = (float8)(-u[i]*mu_1,-u[i]*mu_1,-u[i]*dz*mu_1,-u[i]);
        if (x < Nx-1)
        {
            grad[i].s01 += u[i+1]*mu_1;
        }
        else
        {
            grad[i].s01 = 0.0f;
        }

        if (y < Ny-1)
        {
            grad[i].s23 += u[i+Nx]*mu_1;
        }
        else
        {
            grad[i].s23 = 0.0f;
        }
        if (k < NSl-1)
        {
            grad[i].s45 += u[i+Nx*Ny]*dz*mu_1;
                }
        else
        {
            grad[i].s45 = 0.0f;
        }
        if (uk < NUk-1)
        {
            grad[i].s67 += u[i+NSl*Nx*Ny];
            grad[i].s67 *= dt[uk];
        }
        else
        {
            grad[i].s67 = 0.0f;
        }
        
        i += NSl*Nx*Ny;
    }
}


__kernel void divergence_w_time(
                __global float2 *div,
                __global float8 *p,
                const int NUk,
                const float dz,
                const float mu_1,
                __global float* dt
                )
{
    size_t Nx = get_global_size(2), Ny = get_global_size(1);
    size_t NSl = get_global_size(0);
    size_t x = get_global_id(2), y = get_global_id(1);
    size_t k = get_global_id(0);
    size_t i = k*Nx*Ny+Nx*y + x;
    

    for (int uk=0; uk<NUk; uk++)
    {
        // divergence
        float8 val = p[i];
        if (x == Nx-1)
        {
            val.s01 = 0.0f;
        }
        if (x > 0)
        {   
            val.s01 -= p[i-1].s01;
        }
        if (y == Ny-1)
        {
            val.s23 = 0.0f;
        }
        if (y > 0)
        {
            val.s23 -= p[i-Nx].s23;
        }
        if (k == NSl-1)
        {
            val.s45 = 0.0f;
        }
        if (k > 0)
        {
            val.s45 -= p[i-Nx*Ny].s45;
        }
        if (uk == NUk-1)
        {
            val.s67 = 0.0f;
        }
        else
        {
            val.s67 *= dt[uk];
        }
        if (uk > 0)
        {
            val.s67 -= p[i-NSl*Nx*Ny].s67*dt[uk-1];
        }

        div[i] = (val.s01+val.s23+val.s45*dz)*mu_1 + val.s67;
        // scale gradients
        i += NSl*Nx*Ny;
    }
}


__kernel void update_z2_ictv(
                __global float8 *z_new,
                __global float8 *z,
                __global float8 *gx,
                __global float8 *gx_,
                const float sigma, const float theta, const float alphainv,
                const int NUk
                )
{
    size_t Nx = get_global_size(2), Ny = get_global_size(1);
    size_t NSl = get_global_size(0);
    size_t x = get_global_id(2), y = get_global_id(1);
    size_t k = get_global_id(0);
    size_t i = k*Nx*Ny+Nx*y + x;

    float fac = 0.0f;
    float8 square = 0.0f;

    for (int uk=0; uk<NUk; uk++)
    {
        z_new[i] = z[i] + sigma*((1+theta)*gx[i]-theta*gx_[i]);

        // reproject
        fac = hypot(
                hypot(
                z_new[i].s0,
                z_new[i].s1
                ),
                hypot(
                hypot(
                    z_new[i].s2,
                    z_new[i].s3
                    ),
                hypot(
                    hypot(
                    z_new[i].s4,
                    z_new[i].s5
                    ),
                    hypot(
                    z_new[i].s6,
                    z_new[i].s7
                    )
                    
                )
                )
                );
        fac *= alphainv;
        if (fac > 1.0f){z_new[i] /= fac;}
        i += NSl*Nx*Ny;
    }
}

__kernel void update_z1_ictv(
                __global float8 *z_new,
                __global float8 *z,
                __global float8 *gx1,
                __global float8 *gx1_,
                __global float8 *gx2,
                __global float8 *gx2_,
                const float sigma, const float theta, const float alphainv,
                const int NUk
                )
{
    size_t Nx = get_global_size(2), Ny = get_global_size(1);
    size_t NSl = get_global_size(0);
    size_t x = get_global_id(2), y = get_global_id(1);
    size_t k = get_global_id(0);
    size_t i = k*Nx*Ny+Nx*y + x;

    float fac = 0.0f;
    float8 square = 0.0f;

    for (int uk=0; uk<NUk; uk++)
    {
        z_new[i] = z[i] + sigma*((1+theta)*gx1[i]-theta*gx1_[i]-((1+theta)*gx2[i]-theta*gx2_[i]));

        // reproject
        fac = hypot(
                hypot(
                z_new[i].s0,
                z_new[i].s1
                ),
                hypot(
                hypot(
                    z_new[i].s2,
                    z_new[i].s3
                    ),
                hypot(
                    hypot(
                    z_new[i].s4,
                    z_new[i].s5
                    ),
                    hypot(
                    z_new[i].s6,
                    z_new[i].s7
                    )
                    
                )
                )
                );
        fac *= alphainv;
        if (fac > 1.0f){z_new[i] /= fac;}
        i += NSl*Nx*Ny;
    }
}

__kernel void sym_grad_w_time(
                __global float8 *sym_diag,
                __global float16 *sym_offdiag,
                __global float8 *w,
                const int NUk,
                const float dz,
                const float mu_1,
                __global float* dt
                )
{
    size_t Nx = get_global_size(2), Ny = get_global_size(1);
    size_t NSl = get_global_size(0);
    size_t x = get_global_id(2), y = get_global_id(1);
    size_t k = get_global_id(0);
    size_t i = k*Nx*Ny+Nx*y + x;


    for (int uk=0; uk<NUk; uk++)
    {
        // symmetrized gradient
        float16 val_real = (float16)(
            w[i].s0246, w[i].s0246, w[i].s0246, w[i].s0246);
        float16 val_imag = (float16)(
            w[i].s1357, w[i].s1357, w[i].s1357, w[i].s1357);
        if (x > 0)
        {
            val_real.s0123 -= w[i-1].s0246;
            val_imag.s0123 -= w[i-1].s1357;
        }
        else
        {
            val_real.s0123 = (float4) 0.0f;
            val_imag.s0123 = (float4) 0.0f;
        }

        if (y > 0)
        {
            val_real.s4567 -= w[i-Nx].s0246;
            val_imag.s4567 -= w[i-Nx].s1357;
        }
        else
        {
            val_real.s4567 = (float4) 0.0f;
            val_imag.s4567 = (float4) 0.0f;
        }

        if (k > 0)
        {
            val_real.s89ab -= w[i-Nx*Ny].s0246;
            val_imag.s89ab -= w[i-Nx*Ny].s1357;
        }
        else
        {
            val_real.s89ab = (float4) 0.0f;
            val_imag.s89ab = (float4) 0.0f;
        }
        
        if (uk > 0)
        {
            val_real.scdef -= w[i-Nx*Ny*NSl].s0246;
            val_imag.scdef -= w[i-Nx*Ny*NSl].s1357;
            val_real.scdef *= dt[uk-1];
            val_imag.scdef *= dt[uk-1];  
            
        }
        else
        {
            val_real.scdef = (float4) 0.0f;
            val_imag.scdef = (float4) 0.0f;
        }
        

        sym_diag[i] = (float8)(
          val_real.s0*mu_1, val_imag.s0*mu_1,
          val_real.s5*mu_1, val_imag.s5*mu_1,
          val_real.sa*dz*mu_1, val_imag.sa*dz*mu_1,
          val_real.sf, val_imag.sf);
          
        sym_offdiag[i] = (float16)(
          
          0.5f*(val_real.s1 + val_real.s4)*mu_1,
          0.5f*(val_imag.s1 + val_imag.s4)*mu_1,
          
          0.5f*(val_real.s2 + val_real.s8*dz)*mu_1,
          0.5f*(val_imag.s2 + val_imag.s8*dz)*mu_1,
          
          0.5f*(val_real.s3*mu_1 + val_real.sc),
          0.5f*(val_imag.s3*mu_1 + val_imag.sc),
          
          0.5f*(val_real.s6 + val_real.s9*dz)*mu_1,
          0.5f*(val_imag.s6 + val_imag.s9*dz)*mu_1,
          
          0.5f*(val_real.s7*mu_1 + val_real.sd),
          0.5f*(val_imag.s7*mu_1 + val_imag.sd),
          
          0.5f*(val_real.sb*dz*mu_1 + val_real.se),
          0.5f*(val_imag.sb*dz*mu_1 + val_imag.se),
          
          0.0f, 0.0f, 0.0f, 0.0f
          );
        i += NSl*Nx*Ny;
    }
}


__kernel void sym_divergence_w_time(
                __global float8 *w,
                __global float8 *q_diag,
                __global float16 *q_offdiag,
                const int NUk,
                const float dz,
                const float mu_1,
                __global float* dt
                )
{
    size_t Nx = get_global_size(2), Ny = get_global_size(1);
    size_t NSl = get_global_size(0);
    size_t x = get_global_id(2), y = get_global_id(1);
    size_t k = get_global_id(0);
    size_t i = k*Nx*Ny+Nx*y + x;

    for (int uk=0; uk<NUk; uk++)
    {
        // divergence
        float8 val_diag = -q_diag[i];
        float16 val_offdiag = -q_offdiag[i];
        
        float16 val_real = (float16)(
            val_diag.s0, val_offdiag.s0, val_offdiag.s2, val_offdiag.s4,
            val_offdiag.s0, val_diag.s2, val_offdiag.s6, val_offdiag.s8,
            val_offdiag.s2, val_offdiag.s6, val_diag.s4, val_offdiag.sa,
            val_offdiag.s4, val_offdiag.s8, val_offdiag.sa, val_diag.s6);
            
            
        float16 val_imag = (float16)(
            val_diag.s1, val_offdiag.s1, val_offdiag.s3, val_offdiag.s5,
            val_offdiag.s1, val_diag.s3, val_offdiag.s7, val_offdiag.s9,
            val_offdiag.s3, val_offdiag.s7, val_diag.s5, val_offdiag.sb,
            val_offdiag.s5, val_offdiag.s9, val_offdiag.sb, val_diag.s7);
            
        if (x == 0)
        {
            //real
            val_real.s0123 = 0.0f;
            //imag
            val_imag.s0123 = 0.0f;
        }
        if (x < Nx-1)
        {
            //real
            val_real.s0123 += (float4)(q_diag[i+1].s0, q_offdiag[i+1].s024);
            //imag
            val_imag.s0123 += (float4)(q_diag[i+1].s1, q_offdiag[i+1].s135);
        }
        if (y == 0)
        {
            //real
            val_real.s4567 = 0.0f;
            //imag
            val_imag.s4567 = 0.0f;
        }
        if (y < Ny-1)
        {
            //real
            val_real.s4567 += (float4)(q_offdiag[i+Nx].s0, q_diag[i+Nx].s2, q_offdiag[i+Nx].s68);
            //imag
            val_imag.s4567 += (float4)(q_offdiag[i+Nx].s1, q_diag[i+Nx].s3, q_offdiag[i+Nx].s79);
        }
        if (k == 0)
        {
        //real
            val_real.s89ab = 0.0f;
            //imag
            val_imag.s89ab = 0.0f;
        }
        if (k < NSl-1)
        {
            //real
            val_real.s89ab += (float4)(q_offdiag[i+Nx*Ny].s26, q_diag[i+Nx*Ny].s4, q_offdiag[i+Nx*Ny].sa);
            //imag
            val_imag.s89ab += (float4)(q_offdiag[i+Nx*Ny].s37, q_diag[i+Nx*Ny].s5, q_offdiag[i+Nx*Ny].sb);
        }
        if (uk == 0)
        {
            //real
            val_real.scdef = 0.0f;
            //imag
            val_imag.scdef = 0.0f;
        }
        else
        {
            val_real.scdef *= dt[uk-1];
            val_imag.scdef *= dt[uk-1];
        }
        if (uk < NUk-1)
        {
            //real
            val_real.scdef += (float4)(q_offdiag[i+Nx*Ny*NSl].s48a, q_diag[i+Nx*Ny*NSl].s6)*dt[uk];
            //imag
            val_imag.scdef += (float4)(q_offdiag[i+Nx*Ny*NSl].s59b, q_diag[i+Nx*Ny*NSl].s7)*dt[uk];
        }
        
        // linear step
        //real
        w[i].s0246 = val_real.s0123*mu_1 + val_real.s4567*mu_1 + val_real.s89ab*dz*mu_1 + val_real.scdef;
        //imag
        w[i].s1357 = val_imag.s0123*mu_1 + val_imag.s4567*mu_1 + val_imag.s89ab*dz*mu_1 + val_imag.scdef;

        i += NSl*Nx*Ny;
    }
}


__kernel void operator_fwd_imagerecon(
                __global float2 *out,
                __global float2 *in,
                __global float2 *coils,
                const int NCo,
                const int NScan
                )
{
    size_t X = get_global_size(2);
    size_t Y = get_global_size(1);
    size_t NSl = get_global_size(0);

    size_t x = get_global_id(2);
    size_t y = get_global_id(1);
    size_t k = get_global_id(0);

    float2 tmp_in = 0.0f;
    float2 tmp_coil = 0.0f;


    for (int scan=0; scan<NScan; scan++)
    {
        for (int coil=0; coil < NCo; coil++)
        {
            tmp_coil = coils[coil*NSl*X*Y + k*X*Y + y*X + x];
            tmp_in = in[scan*NSl*X*Y+k*X*Y+ y*X + x];

            out[scan*NCo*NSl*X*Y+coil*NSl*X*Y+k*X*Y + y*X + x] = (float2)(
                        tmp_in.x*tmp_coil.x-tmp_in.y*tmp_coil.y,
                        tmp_in.x*tmp_coil.y+tmp_in.y*tmp_coil.x);
        }
    }
}


__kernel void operator_ad_imagerecon(
                __global float2 *out,
                __global float2 *in,
                __global float2 *coils,
                const int NCo,
                const int NScan
                )
{
    size_t X = get_global_size(2);
    size_t Y = get_global_size(1);
    size_t NSl = get_global_size(0);

    size_t x = get_global_id(2);
    size_t y = get_global_id(1);
    size_t k = get_global_id(0);


    float2 tmp_in = 0.0f;
    float2 conj_coils = 0.0f;


    
    for (int scan=0; scan<NScan; scan++)
    {
        float2 sum = (float2) 0.0f;
        for (int coil=0; coil < NCo; coil++)
        {
            conj_coils = (float2)(
                    coils[coil*NSl*X*Y + k*X*Y + y*X + x].x,
                    -coils[coil*NSl*X*Y + k*X*Y + y*X + x].y);

            tmp_in = in[scan*NCo*NSl*X*Y+coil*NSl*X*Y + k*X*Y+ y*X + x];


            sum += (float2)(
                    tmp_in.x*conj_coils.x-tmp_in.y*conj_coils.y,
                    tmp_in.x*conj_coils.y+tmp_in.y*conj_coils.x);
        }
    
        out[scan*NSl*X*Y+k*X*Y+y*X+x] = sum;
    }
}

__kernel void update_primal_imagerecon(
__global float2 *u_new,
__global float2 *u,
__global float2 *Kyk,
const float tau,
__global float* min,
__global float* max,
__global int* real, const int NUk
)
{
    size_t Nx = get_global_size(2), Ny = get_global_size(1);
    size_t NSl = get_global_size(0);
    size_t x = get_global_id(2), y = get_global_id(1);
    size_t k = get_global_id(0);
    size_t i = k*Nx*Ny+Nx*y + x;
    float norm = 0;
    int idx, idx2, idx3, idx4, idx5;
    float2 tmp;

    for (int uk=0; uk<NUk; uk++)
    {
        u_new[i] = u[i]-tau*Kyk[i];

        if(real[uk]>=1)
        {
            u_new[i].s1 = 0.0f;
            if (u_new[i].s0<min[uk])
            {
                u_new[i].s0 = min[uk];
            }
            if(u_new[i].s0>max[uk])
            {
                u_new[i].s0 = max[uk];
            }
        }
        else
        {
            norm =  sqrt(
              pow(
                (float)(u_new[i].s0),(float)(2.0))
              + pow((float)(u_new[i].s1),(float)(2.0)));
            if (norm<min[uk])
            {
                u_new[i].s0 *= 1/norm*min[uk];
                u_new[i].s1 *= 1/norm*min[uk];
            }
            if(norm>max[uk])
            {
                u_new[i].s0 *= 1/norm*max[uk];
                u_new[i].s1 *= 1/norm*max[uk];
            }
        }
        i += NSl*Nx*Ny;
    }
}

__kernel void update_z2_ictgv(
                __global float8 *z_new,
                __global float8 *z,
                __global float8 *gx,
                __global float8 *gx_,
                __global float8 *w,
                __global float8 *w_,
                const float sigma, const float theta, const float alphainv,
                const int NUk
                )
{
    size_t Nx = get_global_size(2), Ny = get_global_size(1);
    size_t NSl = get_global_size(0);
    size_t x = get_global_id(2), y = get_global_id(1);
    size_t k = get_global_id(0);
    size_t i = k*Nx*Ny+Nx*y + x;

    float fac = 0.0f;
    float8 square = 0.0f;

    for (int uk=0; uk<NUk; uk++)
    {
        z_new[i] = z[i] + sigma*((1+theta)*gx[i]-theta*gx_[i]-((1+theta)*w[i]-theta*w_[i]));

        // reproject
        fac = hypot(
                hypot(
                z_new[i].s0,
                z_new[i].s1
                ),
                hypot(
                hypot(
                    z_new[i].s2,
                    z_new[i].s3
                    ),
                hypot(
                    hypot(
                    z_new[i].s4,
                    z_new[i].s5
                    ),
                    hypot(
                    z_new[i].s6,
                    z_new[i].s7
                    )
                    
                )
                )
                );
        fac *= alphainv;
        if (fac > 1.0f){z_new[i] /= fac;}
        i += NSl*Nx*Ny;
    }
}

__kernel void update_z1_ictgv(
                __global float8 *z_new,
                __global float8 *z,
                __global float8 *gx1,
                __global float8 *gx1_,
                __global float8 *gx2,
                __global float8 *gx2_,
                __global float8 *w,
                __global float8 *w_,
                const float sigma, const float theta, const float alphainv,
                const int NUk
                )
{
    size_t Nx = get_global_size(2), Ny = get_global_size(1);
    size_t NSl = get_global_size(0);
    size_t x = get_global_id(2), y = get_global_id(1);
    size_t k = get_global_id(0);
    size_t i = k*Nx*Ny+Nx*y + x;

    float fac = 0.0f;
    float8 square = 0.0f;

    for (int uk=0; uk<NUk; uk++)
    {
        z_new[i] = z[i] + sigma*((1+theta)*gx1[i]-theta*gx1_[i]-((1+theta)*gx2[i]-theta*gx2_[i])-((1+theta)*w[i]-theta*w_[i]));

        // reproject
        fac = hypot(
                hypot(
                z_new[i].s0,
                z_new[i].s1
                ),
                hypot(
                hypot(
                    z_new[i].s2,
                    z_new[i].s3
                    ),
                hypot(
                    hypot(
                    z_new[i].s4,
                    z_new[i].s5
                    ),
                    hypot(
                    z_new[i].s6,
                    z_new[i].s7
                    )
                    
                )
                )
                );
        fac *= alphainv;
        if (fac > 1.0f){z_new[i] /= fac;}
        i += NSl*Nx*Ny;
    }
}

__kernel void update_z3_ictgv(
                __global float8 *z_new_diag,
                __global float16 *z_new_offdiag,
                __global float8 *z_diag,
                __global float16 *z_offdiag,
                __global float8 *gx_diag,
                __global float16 *gx_offdiag,
                __global float8 *gx_diag_,
                __global float16 *gx_offdiag_,
                const float sigma,
                const float theta,
                const float alphainv,
                const int NUk
                )
{
    size_t Nx = get_global_size(2), Ny = get_global_size(1);
    size_t NSl = get_global_size(0);
    size_t x = get_global_id(2), y = get_global_id(1);
    size_t k = get_global_id(0);
    size_t i = k*Nx*Ny+Nx*y + x;

    float fac = 0.0f;
    float16 square = 0.0f;

    for (int uk=0; uk<NUk; uk++)
    {
        z_new_diag[i] = z_diag[i] + sigma*((1+theta)*gx_diag[i]-theta*gx_diag_[i]);
        z_new_offdiag[i] = z_offdiag[i] + sigma*((1+theta)*gx_offdiag[i]-theta*gx_offdiag_[i]);

        // reproject
        fac = hypot(fac,
        hypot(
            hypot(
                hypot(
                    hypot(
                      z_new_diag[i].s0,
                      z_new_diag[i].s1
                      ),
                  hypot(
                    z_new_diag[i].s2,
                    z_new_diag[i].s3
                    )
                ),
                hypot(
                    hypot(
                      z_new_diag[i].s4,
                      z_new_diag[i].s5
                      ),
                    hypot(
                      z_new_diag[i].s6,
                      z_new_diag[i].s7
                      )
                )
            ),
            hypot(
                hypot(
                    hypot(
                      2.0f*hypot(
                        z_new_offdiag[i].s0,
                        z_new_offdiag[i].s1
                        ),
                      2.0f*hypot(
                        z_new_offdiag[i].s2,
                        z_new_offdiag[i].s3
                        )
                    ),
                    hypot(
                      2.0f*hypot(
                        z_new_offdiag[i].s4,
                        z_new_offdiag[i].s5
                        ),
                      2.0f*hypot(
                        z_new_offdiag[i].s6,
                        z_new_offdiag[i].s7
                        )
                    )
                ),
                hypot(
                  2.0f*hypot(
                    z_new_offdiag[i].s8,
                    z_new_offdiag[i].s9
                    ),
                  2.0f*hypot(
                    z_new_offdiag[i].sa,
                    z_new_offdiag[i].sb
                    )
                )                   
            )
          )
        );

        i += NSl*Nx*Ny;
    }
    fac *= alphainv;
    i = k*Nx*Ny+Nx*y + x;
    for (int uk=0; uk<NUk; uk++)
    {
        if (fac > 1.0f){
            z_new_diag[i] /= fac;
            z_new_offdiag[i] /= fac;
        }
        i += NSl*Nx*Ny;
    }
}