__kernel void update_v(
                __global float8 *v,
                __global float8 *v_,
                __global float8 *Kyk2,
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
                const int dimZ,
                const int dimY,
                const int dimX
                )
{
    size_t i = get_global_id(0);
    size_t size = get_global_size(0);
    
    size_t ind = 0;
    size_t ind_slice = i/(dimX*dimY);
    size_t ind_xy = i%(dimX*dimY);
    
    float fac = 0.0f;

    for (int uk=0; uk<NUk; uk++)
    {
        ind = ind_slice*(dimX*dimY*NUk) + uk*dimX*dimY + ind_xy;
        z_new[ind] = z[ind] + sigma*((1+theta)*gx[ind]-theta*gx_[ind]);

        // reproject
        fac = hypot(fac,
        hypot(
          hypot(
            hypot(
              hypot(
                z_new[ind].s0,
                z_new[ind].s1
                ),
                hypot(
                  z_new[ind].s2,
                  z_new[ind].s3
                  )
              ),
              hypot(
                z_new[ind].s4
                ,z_new[ind].s5
                )
            ),
            hypot(
              hypot(
                2.0f*hypot(
                  z_new[ind].s6,
                  z_new[ind].s7
                  ),
                2.0f*hypot(
                  z_new[ind].s8,
                  z_new[ind].s9
                  )
                ),
                2.0f*hypot(
                  z_new[ind].sa,
                  z_new[ind].sb
                  )
              )
            )
          );
    }
    fac *= alphainv;

    for (int uk=0; uk<NUk; uk++)
    {
        ind = ind_slice*(dimX*dimY*NUk) + uk*dimX*dimY + ind_xy;
        if (fac > 1.0f) z_new[ind] /=fac;
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
                const int dimZ,
                const int dimY,
                const int dimX,
                const float h1inv
                )
{
    size_t i = get_global_id(0);
    size_t size = get_global_size(0);
    
    size_t ind = 0;
    size_t ind_slice = i/(dimX*dimY);
    size_t ind_xy = i%(dimX*dimY);
    
    size_t NUk = NUk_tgv+NUk_H1;

    float fac = 0.0f;
    
    

    for (int uk=0; uk<NUk_tgv; uk++)
    {
       ind = ind_slice*(dimX*dimY*NUk) + uk*dimX*dimY + ind_xy;
       z_new[ind] = z[ind] + sigma*(
           (1+theta)*gx[ind]-theta*gx_[ind]-(1+theta)*vx[ind]+theta*vx_[ind]);

       // reproject
       fac = hypot(fac,
       hypot(
         hypot(
           z_new[ind].s0,
           z_new[ind].s1
           ),
         hypot(
           hypot(
             z_new[ind].s2,
             z_new[ind].s3
             ),
          hypot(
            z_new[ind].s4,
            z_new[ind].s5
            )
          )
        )
       );
    }
    fac *= alphainv;
    for (int uk=0; uk<NUk_tgv; uk++)
    {
        ind = ind_slice*(dimX*dimY*NUk) + uk*dimX*dimY + ind_xy;
        if (fac > 1.0f) z_new[ind] /=fac;
    }
    for (int uk=NUk_tgv; uk<(NUk_tgv+NUk_H1); uk++)
    {
        ind = ind_slice*(dimX*dimY*NUk) + uk*dimX*dimY + ind_xy;
        z_new[ind] = (z[ind] + sigma*((1+theta)*gx[ind]-theta*gx_[ind]))*h1inv;
    }
}


__kernel void update_z1_tv(
                __global float8 *z_new,
                __global float8 *z,
                __global float8 *gx,
                __global float8 *gx_,
                const float sigma,
                const float theta,
                const float alphainv,
                const int NUk_tgv,
                const int NUk_H1,
                const int dimZ,
                const int dimY,
                const int dimX,
                const float h1inv
                )
{
    size_t i = get_global_id(0);
    size_t size = get_global_size(0);
    
    size_t ind = 0;
    size_t ind_slice = i/(dimX*dimY);
    size_t ind_xy = i%(dimX*dimY);
    
    size_t NUk = NUk_tgv+NUk_H1;

    float fac = 0.0f;

    for (int uk=0; uk<NUk_tgv; uk++)
    {
        ind = ind_slice*(dimX*dimY*NUk) + uk*dimX*dimY + ind_xy;
        z_new[ind] = z[ind] + sigma*((1+theta)*gx[ind]-theta*gx_[ind]);

        // reproject
        //square = powr(z_new[ind], 2);
        //fac += square.s0+square.s1+square.s2+square.s3+square.s4+square.s5;
        fac = hypot(fac,
        hypot(
          hypot(
            z_new[ind].s0,
            z_new[ind].s1
            ),
          hypot(
            hypot(
              z_new[ind].s2,
              z_new[ind].s3
              ),
            hypot(
              z_new[ind].s4,
              z_new[ind].s5
              )
            )
          )
        );
    }
    fac *= alphainv;
    for (int uk=0; uk<NUk_tgv; uk++)
    {   
        ind = ind_slice*(dimX*dimY*NUk) + uk*dimX*dimY + ind_xy;
        if (fac > 1.0f){z_new[ind] /= fac;}
    }
    for (int uk=NUk_tgv; uk<(NUk_tgv+NUk_H1); uk++)
    {
        ind = ind_slice*(dimX*dimY*NUk) + uk*dimX*dimY + ind_xy;
        z_new[ind] = (z[ind] + sigma*((1+theta)*gx[ind]-theta*gx_[ind]))*h1inv;
    }
}


__kernel void update_primal(
                __global float2 *u_new,
                __global float2 *u,
                __global float2 *Kyk,
                __global float2 *u_k,
                const float tau,
                const float tauinv,
                const float div,
                __global float* min,
                __global float* max,
                __global int* real, 
                const int NUk,
                const int dimZ,
                const int dimY,
                const int dimX
                )
{
    size_t i = get_global_id(0);
    size_t size = get_global_size(0);
    float norm = 0;

    u_new[i] = (u[i]-tau*Kyk[i]+tauinv*u_k[i])*div;

    if(real[(i/(dimX*dimY))%NUk]>=1)
    {
        u_new[i].s1 = 0.0f;
        if (u_new[i].s0<min[(i/(dimX*dimY))%NUk])
        {
            u_new[i].s0 = min[(i/(dimX*dimY))%NUk];
        }
        if(u_new[i].s0>max[(i/(dimX*dimY))%NUk])
        {
            u_new[i].s0 = max[(i/(dimX*dimY))%NUk];
        }
    }
    else
    {
        norm =  sqrt(
          pow(
            (float)(u_new[i].s0),(float)(2.0))
          + pow((float)(u_new[i].s1),(float)(2.0)));
        if (norm<min[(i/(dimX*dimY))%NUk])
        {
            u_new[i].s0 *= 1/norm*min[(i/(dimX*dimY))%NUk];
            u_new[i].s1 *= 1/norm*min[(i/(dimX*dimY))%NUk];
        }
        if(norm>max[(i/(dimX*dimY))%NUk])
        {
            u_new[i].s0 *= 1/norm*max[(i/(dimX*dimY))%NUk];
            u_new[i].s1 *= 1/norm*max[(i/(dimX*dimY))%NUk];
        }
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
                const int dimZ,
                const int dimY,
                const int dimX
                )
{
    size_t i = get_global_id(0);
    size_t size = get_global_size(0);
    float norm = 0;

    u_new[i] = (u[i]-tau*Kyk[i]+tauinv*A[i]*u_k[i])/(1+tauinv*A[i]);

    if(real[(i/(dimX*dimY))%NUk]>=1)
    {
        u_new[i].s1 = 0.0f;
        if (u_new[i].s0<min[(i/(dimX*dimY))%NUk])
        {
            u_new[i].s0 = min[(i/(dimX*dimY))%NUk];
        }
        if(u_new[i].s0>max[(i/(dimX*dimY))%NUk])
        {
            u_new[i].s0 = max[(i/(dimX*dimY))%NUk];
        }
    }
    else
    {
        norm =  sqrt(
                  pow((float)(u_new[i].s0),(float)(2.0))
                  + pow((float)(u_new[i].s1),(float)(2.0)));
        if (norm<min[(i/(dimX*dimY))%NUk])
        {
            u_new[i].s0 *= 1/norm*min[(i/(dimX*dimY))%NUk];
            u_new[i].s1 *= 1/norm*min[(i/(dimX*dimY))%NUk];
        }
        if(norm>max[(i/(dimX*dimY))%NUk])
        {
            u_new[i].s0 *= 1/norm*max[(i/(dimX*dimY))%NUk];
            u_new[i].s1 *= 1/norm*max[(i/(dimX*dimY))%NUk];
        }
    }
}

__kernel void gradient(
                __global float8 *grad,
                __global float2 *u,
                const int NUk,
                const int dimZ,
                const int dimY,
                const int dimX,
                __global float* ratio,
                const float dz
                )
{
    size_t i = get_global_id(0);
    size_t size = get_global_size(0);

    // gradient
    float2 back_point = u[i];
    float8 mygrad = 0.0f;
    
    mygrad.s01 = ( (i+1) % dimX) ? (u[i+1]-back_point) : 0.0f;
    mygrad.s23 = ( (i/dimX + 1) % dimY) ? (u[i+dimX]-back_point) : 0.0f;
    mygrad.s45 = ( (i/(dimX*dimY*NUk) + 1) % dimZ) ? (u[i+dimX*dimY*NUk]-back_point) * dz : 0.0f;
    
    // scale gradients
    mygrad*=ratio[(i/(dimX*dimY))%NUk];
    grad[i] = mygrad;
}


__kernel void sym_grad(
                __global float16 *sym,
                __global float8 *w,
                const int NUk,
                const int dimZ,
                const int dimY,
                const int dimX,
                __global float* ratio,
                const float dz
                )
{
    size_t i = get_global_id(0);
    size_t size = get_global_size(0);
    
    float3 val_r_tmp = w[i].s024;
    float3 val_i_tmp = w[i].s135;
        
    float3 real_xdiff=val_r_tmp;
    float3 imag_xdiff=val_i_tmp;
    
    float3 real_ydiff=val_r_tmp;
    float3 imag_ydiff=val_i_tmp;
    
    float3 real_zdiff=val_r_tmp;
    float3 imag_zdiff=val_i_tmp;

    real_xdiff = ( i % dimX) ? (real_xdiff - w[i-1].s024) : 0.0f;
    imag_xdiff = ( i % dimX) ? (imag_xdiff - w[i-1].s135) : 0.0f;

    real_ydiff = ( (i/dimX) % dimY) ? (real_ydiff - w[i-dimX].s024) : 0.0f;
    imag_ydiff = ( (i/dimX) % dimY) ? (imag_ydiff - w[i-dimX].s135) : 0.0f;
    
    real_zdiff = ( (i/(dimX*dimY*NUk)) % dimZ) ? (real_zdiff - w[i-dimX*dimY*NUk].s024) : 0.0f;
    imag_zdiff = ( (i/(dimX*dimY*NUk)) % dimZ) ? (imag_zdiff - w[i-dimX*dimY*NUk].s135) : 0.0f;
    
    sym[i] = (float16)(
      real_xdiff.s0, imag_xdiff.s0,
      real_ydiff.s1, imag_ydiff.s1,
      real_zdiff.s2*dz, imag_zdiff.s2*dz,
      0.5f*(real_xdiff.s1 + real_ydiff.s0),
      0.5f*(imag_xdiff.s1 + imag_ydiff.s0),
      0.5f*(real_xdiff.s2 + real_zdiff.s0*dz),
      0.5f*(imag_xdiff.s2 + imag_zdiff.s0*dz),
      0.5f*(real_ydiff.s2 + real_zdiff.s1*dz),
      0.5f*(imag_ydiff.s2 + imag_zdiff.s1*dz),
      0.0f,0.0f,0.0f,0.0f)*ratio[(i/(dimX*dimY))%NUk];    
}


__kernel void divergence(
                __global float2 *div,
                __global float8 *p,
                const int NUk,
                const int dimZ,
                const int dimY,
                const int dimX,
                __global float* ratio,
                const int last,
                const float dz
                )
{
    size_t i = get_global_id(0);
    size_t size = get_global_size(0);

    float8 val = p[i];
    
    if (!((i+1)%dimX))
      val.s01 = 0.0f;
    val.s01 = ( i % dimX) ? (val.s01 - p[i-1].s01) : val.s01;

    if (!((i/dimX+1)%dimY))
      val.s23 = 0.0f;
    val.s23 = ( (i/dimX) % dimY) ? (val.s23 - p[i-dimX].s23) : val.s23;
    
    if (last == 1)
    {
        if (!((i/(dimX*dimY*NUk)+1)%dimZ))
          val.s45 = 0.0f;
    }
    val.s45 = ( (i/(dimX*dimY*NUk)) % dimZ) ? (val.s45 - p[i-dimX*dimY*NUk].s45) * dz : val.s45 * dz;

    div[i] = (val.s01+val.s23+val.s45)*ratio[(i/(dimX*dimY))%NUk];

}


__kernel void sym_divergence(
                __global float8 *w,
                __global float16 *q,
                const int NUk,
                const int dimZ,
                const int dimY,
                const int dimX,
                __global float* ratio,
                const int first,
                const float dz
                )
{
    size_t i = get_global_id(0);
    size_t size = get_global_size(0);
    
    float16 val0 = -q[i];
           
    float3 real_xdiff=val0.s068;
    float3 imag_xdiff=val0.s179;
    
    float3 real_ydiff=val0.s62a;
    float3 imag_ydiff=val0.s73b;
    
    float3 real_zdiff=val0.s8a4;
    float3 imag_zdiff=val0.s9b5;

    if (!(i % dimX))
    {
       real_xdiff = 0.0f;
       imag_xdiff = 0.0f;
    }
      
    real_xdiff = ( (i+1) % dimX) ? (real_xdiff + q[i+1].s068) : real_xdiff;
    imag_xdiff = ( (i+1) % dimX) ? (imag_xdiff + q[i+1].s179) : imag_xdiff;
    
    if (!((i/dimX) % dimY))
    {
       real_ydiff = 0.0f;
       imag_ydiff = 0.0f;
    }
    real_ydiff = ( (i/dimX + 1) % dimY) ? (real_ydiff + q[i+dimX].s62a) : real_ydiff;
    imag_ydiff = ( (i/dimX + 1) % dimY) ? (imag_ydiff + q[i+dimX].s73b) : imag_ydiff;
    
    if (first == 1)
    {
        if (!((i/(dimX*dimY*NUk)) % dimZ))
        {
           real_zdiff = 0.0f;
           imag_zdiff = 0.0f;
        }
    }
    real_zdiff = ( (i/(dimX*dimY*NUk) + 1) % dimZ) ? (real_zdiff + q[i+dimX*dimY*NUk].s8a4) : real_zdiff;
    imag_zdiff = ( (i/(dimX*dimY*NUk) + 1) % dimZ) ? (imag_zdiff + q[i+dimX*dimY*NUk].s9b5) : imag_zdiff;
        
    w[i].s024 = (real_xdiff + real_ydiff + real_zdiff*dz)*ratio[(i/(dimX*dimY))%NUk];
    w[i].s135 = (imag_xdiff + imag_ydiff + imag_zdiff*dz)*ratio[(i/(dimX*dimY))%NUk];
}


__kernel void update_Kyk2(
                __global float8 *w,
                __global float16 *q,
                __global float8 *z,
                const int NUk,
                const int dimZ,
                const int dimY,
                const int dimX,
                __global float* ratio,
                const int first,
                const float dz
                )
{
    size_t i = get_global_id(0);
    size_t size = get_global_size(0);

    float16 val0 = -q[i];
           
    float3 real_xdiff=val0.s068;
    float3 imag_xdiff=val0.s179;
    
    float3 real_ydiff=val0.s62a;
    float3 imag_ydiff=val0.s73b;
    
    float3 real_zdiff=val0.s8a4;
    float3 imag_zdiff=val0.s9b5;

    if (!(i % dimX))
    {
       real_xdiff = 0.0f;
       imag_xdiff = 0.0f;
    }
      
    real_xdiff = ( (i+1) % dimX) ? (real_xdiff + q[i+1].s068) : real_xdiff;
    imag_xdiff = ( (i+1) % dimX) ? (imag_xdiff + q[i+1].s179) : imag_xdiff;
    
    if (!((i/dimX) % dimY))
    {
       real_ydiff = 0.0f;
       imag_ydiff = 0.0f;
    }
    real_ydiff = ( (i/dimX + 1) % dimY) ? (real_ydiff + q[i+dimX].s62a) : real_ydiff;
    imag_ydiff = ( (i/dimX + 1) % dimY) ? (imag_ydiff + q[i+dimX].s73b) : imag_ydiff;
    
    if (first == 1)
    {
        if (!((i/(dimX*dimY*NUk)) % dimZ))
        {
           real_zdiff = 0.0f;
           imag_zdiff = 0.0f;
        }
    }
    real_zdiff = ( (i/(dimX*dimY*NUk) + 1) % dimZ) ? (real_zdiff + q[i+dimX*dimY*NUk].s8a4) : real_zdiff;
    imag_zdiff = ( (i/(dimX*dimY*NUk) + 1) % dimZ) ? (imag_zdiff + q[i+dimX*dimY*NUk].s9b5) : imag_zdiff;
    
    w[i].s024 = -(real_xdiff + real_ydiff + real_zdiff*dz)*ratio[(i/(dimX*dimY))%NUk]-z[i].s024;
    w[i].s135 = -(imag_xdiff + imag_ydiff + imag_zdiff*dz)*ratio[(i/(dimX*dimY))%NUk]-z[i].s135;
}

__kernel void operator_fwd(
                __global float2 *out,
                __global float2 *in,
                __global float2 *coils,
                __global float2 *grad,
                const int NCo,
                const int NScan,
                const int NUk,
                const int dimZ,
                const int dimY,
                const int dimX
                )
{
    size_t i = get_global_id(0);
    size_t size = get_global_size(0);
    
    size_t ind = 0;
    size_t ind_slice = i/(dimX*dimY);
    size_t ind_xy = i%(dimX*dimY);
    size_t XY = dimX*dimY;

    float2 tmp_in = 0.0f;
    float2 tmp_grad = 0.0f;
    float2 tmp_coil = 0.0f;
    float2 tmp_mul = 0.0f;


    for (int scan=0; scan<NScan; scan++)
    {
        for (int coil=0; coil < NCo; coil++)
        {
            tmp_coil = coils[ind_slice*NCo*XY + coil*XY + ind_xy];
            float2 sum = 0.0f;
            for (int uk=0; uk<NUk; uk++)
            {
                tmp_grad = grad[ind_slice*NUk*NScan*XY
                                +uk*NScan*XY+scan*XY + ind_xy];
                tmp_in = in[ind_slice*NUk*XY+uk*XY + ind_xy];

                tmp_mul = (float2)(tmp_in.x*tmp_grad.x-tmp_in.y*tmp_grad.y,
                                   tmp_in.x*tmp_grad.y+tmp_in.y*tmp_grad.x
                                  );
                sum += (float2)(tmp_mul.x*tmp_coil.x-tmp_mul.y*tmp_coil.y,
                                tmp_mul.x*tmp_coil.y+tmp_mul.y*tmp_coil.x);

            }
            out[ind_slice*NScan*NCo*XY+scan*NCo*XY+coil*XY + ind_xy] = sum;
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
                const int NUk,
                const int dimZ,
                const int dimY,
                const int dimX
                )
{
    size_t i = get_global_id(0);
    size_t size = get_global_size(0);
    
    size_t ind = 0;
    size_t ind_slice = i/(dimX*dimY);
    size_t ind_xy = i%(dimX*dimY);
    size_t XY = dimX*dimY;



    float2 tmp_in = 0.0f;
    float2 tmp_mul = 0.0f;
    float2 conj_grad = 0.0f;
    float2 conj_coils = 0.0f;


    for (int uk=0; uk<NUk; uk++)
    {
        float2 sum = (float2) 0.0f;
        for (int scan=0; scan<NScan; scan++)
        {
            conj_grad = (float2) (
                grad[ind_slice*NUk*NScan*XY+uk*NScan*XY+scan*XY + ind_xy].x,
                - grad[ind_slice*NUk*NScan*XY+uk*NScan*XY+scan*XY + ind_xy].y
                );
            for (int coil=0; coil < NCo; coil++)
            {
                conj_coils = (float2) (
                    coils[ind_slice*NCo*XY + coil*XY + ind_xy].x,
                    - coils[ind_slice*NCo*XY + coil*XY + ind_xy].y);

                tmp_in = in[ind_slice*NScan*NCo*XY+scan*NCo*XY+coil*XY + ind_xy];
                tmp_mul = (float2)(tmp_in.x*conj_grad.x-tmp_in.y*conj_grad.y,
                                   tmp_in.x*conj_grad.y+tmp_in.y*conj_grad.x
                                  );


                sum += (float2)(tmp_mul.x*conj_coils.x-tmp_mul.y*conj_coils.y,
                                tmp_mul.x*conj_coils.y+tmp_mul.y*conj_coils.x
                               );
            }
        }
        out[ind_slice*NUk*XY+uk*XY+ind_xy] = sum;
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
                __global float* ratio,
                const int NUk,
                const int dimZ,
                const int dimY,
                const int dimX,
                const int last,
                const float dz
                )
{
    size_t i = get_global_id(0);
    size_t size = get_global_size(0);
    
    size_t ind = 0;
    size_t ind_slice = i/(dimX*dimY);
    size_t ind_xy = i%(dimX*dimY);
    size_t XY = dimX*dimY;


    float2 tmp_in = 0.0f;
    float2 tmp_mul = 0.0f;
    float2 conj_grad = 0.0f;
    float2 conj_coils = 0.0f;



    for (int uk=0; uk<NUk; uk++)
    {
        float2 sum = (float2) 0.0f;
        for (int scan=0; scan<NScan; scan++)
        {
            conj_grad = (float2) (
                grad[ind_slice*NUk*NScan*XY+uk*NScan*XY+scan*XY + ind_xy].x,
                - grad[ind_slice*NUk*NScan*XY+uk*NScan*XY+scan*XY + ind_xy].y
                );
            for (int coil=0; coil < NCo; coil++)
            {
                conj_coils = (float2) (
                    coils[ind_slice*NCo*XY + coil*XY + ind_xy].x,
                    -coils[ind_slice*NCo*XY + coil*XY + ind_xy].y
                    );

                tmp_in = in[ind_slice*NScan*NCo*XY+scan*NCo*XY+coil*XY + ind_xy];
                tmp_mul = (float2)(tmp_in.x*conj_grad.x-tmp_in.y*conj_grad.y,
                                   tmp_in.x*conj_grad.y+tmp_in.y*conj_grad.x
                                  );


                sum += (float2)(tmp_mul.x*conj_coils.x-tmp_mul.y*conj_coils.y,
                                tmp_mul.x*conj_coils.y+tmp_mul.y*conj_coils.x
                               );
            }
        }

        // divergence
        ind = ind_slice*NUk*XY+uk*XY+ind_xy;
        float8 val = p[ind];
        
        if (!((ind+1)%dimX))
          val.s01 = 0.0f;
        val.s01 = ( ind % dimX) ? (val.s01 - p[ind-1].s01) : val.s01;
    
        if (!((ind/dimX+1)%dimY))
          val.s23 = 0.0f;
        val.s23 = ( (ind/dimX) % dimY) ? (val.s23 - p[i-dimX].s23) : val.s23;
        
        if (last == 1)
        {
            if (!((ind/(dimX*dimY*NUk)+1)%dimZ))
              val.s45 = 0.0f;
        }
        val.s45 = ( (ind/(dimX*dimY*NUk)) % dimZ) ? (val.s45 - p[ind-dimX*dimY*NUk].s45) * dz : val.s45 * dz;
        // scale gradients
        val*=ratio[uk];
        
        out[ind] = sum - (val.s01+val.s23+val.s45);
    }
}


__kernel void update_Kyk1SMS(
                __global float2 *out,
                __global float2 *in,
                __global float8 *p,
                __global float* ratio,
                const int NUk,
                const int dimZ,
                const int dimY,
                const int dimX,
                const int last,
                const float dz
                )
{
    size_t i = get_global_id(0);
    size_t size = get_global_size(0);
    
    size_t ind = 0;
    size_t ind_slice = i/(dimX*dimY);
    size_t ind_xy = i%(dimX*dimY);
    size_t XY = dimX*dimY;

    for (int uk=0; uk<NUk; uk++)
    {
        ind = ind_slice*NUk*XY+uk*XY+ind_xy;
        float8 val = p[ind];
        
        if (!((ind+1)%dimX))
          val.s01 = 0.0f;
        val.s01 = ( ind % dimX) ? (val.s01 - p[ind-1].s01) : val.s01;
    
        if (!((ind/dimX+1)%dimY))
          val.s23 = 0.0f;
        val.s23 = ( (ind/dimX) % dimY) ? (val.s23 - p[ind-dimX].s23) : val.s23;
        
        if (last == 1)
        {
            if (!((ind/(dimX*dimY*NUk)+1)%dimZ))
              val.s45 = 0.0f;
        }
        val.s45 = ( (ind/(dimX*dimY*NUk)) % dimZ) ? (val.s45 - p[ind-dimX*dimY*NUk].s45) * dz : val.s45 * dz;
        // scale gradients
        val*=ratio[uk];
        
        out[ind] = in[ind] - (val.s01+val.s23+val.s45);
    }
}


__kernel void operator_fwd_imagespace(
                __global float2 *out,
                __global float2 *in,
                __global float2 *grad,
                const int NScan,
                const int NUk,
                const int dimZ,
                const int dimY,
                const int dimX
                )
{
    size_t i = get_global_id(0);
    size_t size = get_global_size(0);
    
    size_t ind = 0;
    size_t ind_slice = i/(dimX*dimY);
    size_t ind_xy = i%(dimX*dimY);
    size_t XY = dimX*dimY;

    float2 tmp_in = 0.0f;
    float2 tmp_grad = 0.0f;

    for (int scan=0; scan<NScan; scan++)
    {
        float2 sum = 0.0f;
        for (int uk=0; uk<NUk; uk++)
        {
            tmp_grad = grad[ind_slice*NUk*NScan*XY
                            +uk*NScan*XY+ scan*XY + ind_xy];
            tmp_in = in[ind_slice*NUk*XY+ uk*XY + ind_xy];
    
          sum += (float2)(tmp_in.x*tmp_grad.x-tmp_in.y*tmp_grad.y,tmp_in.x*tmp_grad.y+tmp_in.y*tmp_grad.x);
    
        }
        out[ind_slice*NScan*XY+scan*XY+ind_xy] = sum;
    }
}

__kernel void operator_ad_imagespace(
                __global float2 *out, 
                __global float2 *in,
                __global float2 *grad,
                const int NScan, 
                const int NUk,
                const int dimZ,
                const int dimY,
                const int dimX
    )
{
    size_t i = get_global_id(0);
    size_t size = get_global_size(0);
    
    size_t ind = 0;
    size_t ind_slice = i/(dimX*dimY);
    size_t ind_xy = i%(dimX*dimY);
    size_t XY = dimX*dimY;

    float2 tmp_in = 0.0f;
    float2 conj_grad = 0.0f;

    for (int uk=0; uk<NUk; uk++)
    {
        float2 sum = (float2) 0.0f;
        for (int scan=0; scan<NScan; scan++)
        {
            conj_grad = (float2) (
                grad[ind_slice*NUk*NScan*XY+uk*NScan*XY+scan*XY + ind_xy].x,
                - grad[ind_slice*NUk*NScan*XY+uk*NScan*XY+scan*XY + ind_xy].y
                );
            tmp_in = in[ind_slice*NScan*XY + scan*XY+ ind_xy];
            sum += (float2)(tmp_in.x*conj_grad.x-tmp_in.y*conj_grad.y,
                            tmp_in.x*conj_grad.y+tmp_in.y*conj_grad.x
                           );
        }
        out[ind_slice*NUk*XY+uk*XY+ind_xy] = sum;
    }
}


__kernel void update_Kyk1_imagespace(
                __global float2 *out,
                __global float2 *in,
                __global float2 *grad,
                __global float8 *p,
                const int NScan,
                __global float* ratio,
                const int NUk,
                const int dimZ,
                const int dimY,
                const int dimX,
                const int last,
                const float dz
                )
{
    size_t i = get_global_id(0);
    size_t size = get_global_size(0);
    
    size_t ind = 0;
    size_t ind_slice = i/(dimX*dimY);
    size_t ind_xy = i%(dimX*dimY);
    size_t XY = dimX*dimY;

    float2 tmp_in = 0.0f;
    float2 conj_grad = 0.0f;

    for (int uk=0; uk<NUk; uk++)
    {
        float2 sum = (float2) 0.0f;
        for (int scan=0; scan<NScan; scan++)
        {
            conj_grad = (float2) (
                grad[ind_slice*NUk*NScan*XY+uk*NScan*XY+scan*XY + ind_xy].x,
                - grad[ind_slice*NUk*NScan*XY+uk*NScan*XY+scan*XY + ind_xy].y
                );
            tmp_in = in[ind_slice*NScan*XY + scan*XY+ ind_xy];
            sum += (float2)(tmp_in.x*conj_grad.x-tmp_in.y*conj_grad.y,
                            tmp_in.x*conj_grad.y+tmp_in.y*conj_grad.x
                           );
        }

        // divergence
        ind = ind_slice*NUk*XY+uk*XY+ind_xy;
        float8 val = p[ind];
        
        if (!((ind+1)%dimX))
          val.s01 = 0.0f;
        val.s01 = ( ind % dimX) ? (val.s01 - p[ind-1].s01) : val.s01;
    
        if (!((ind/dimX+1)%dimY))
          val.s23 = 0.0f;
        val.s23 = ( (ind/dimX) % dimY) ? (val.s23 - p[ind-dimX].s23) : val.s23;
        
        if (last == 1)
        {
            if (!((ind/(dimX*dimY*NUk)+1)%dimZ))
              val.s45 = 0.0f;
        }
        val.s45 = ( (ind/(dimX*dimY*NUk)) % dimZ) ? (val.s45 - p[ind-dimX*dimY*NUk].s45) * dz : val.s45 * dz;
        // scale gradients
        val*=ratio[uk];
        
        out[ind] = sum - (val.s01+val.s23+val.s45);
    }
}


__kernel void perumtescansslices(
                __global float2 *out,
                __global float2 *in
                )
{
    size_t XY = get_global_size(2);
    size_t NSl = get_global_size(1);
    size_t ScanCoil = get_global_size(0);
    size_t xy = get_global_id(2);
    size_t sl = get_global_id(1);
    size_t sc = get_global_id(0);

    out[xy + XY*sc + XY*ScanCoil*sl] = in[xy + XY*sl + XY*NSl*sc];
}
