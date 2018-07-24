
def imax(x1,x2):
# 	Returns maximum of two numbers.
#
  retval = x1
  if (x2>x1):
    	retval = x2

  printf("Max is %d\n",retval)
  return retval




def convkernalpyr(u,v,gridspace):

#      Evaluates Convolution kernal at (u,v).
#      Pyramid Kernal.
#

double u;
double v;
double gridspace;

{
double retval = 0;

u = fabs(u)/gridspace;
v = fabs(v)/gridspace;


if ((u < 1.0) && (v< 1.0))
        retval = (1-u) * (1-v);
else
        retval = 0.0;

return retval;
}



double convkernalgauss(u,v,width)

#	Evaluates Convolution kernal at (u,v).	#

double u;
double v;
double width;

{
double retval = 0;
double d;
double sig;

d = (u*u+v*v)/width/width;

if (d < 1.0)
	{
	sig = (0.0969 + 0.0707*width);
	retval = exp(-d/sig);
	}
else
	retval = 0.0;

return retval;
}





double convkernal(u,v,width)

#      Evaluates Convolution kernal at (u,v).  #

double u;
double v;
double width;

{
double retval = 0;
double d;
double sig;

d = (u*u+v*v)/width/width;


if (d < 1.0)
        retval = (1-sqrt(d));
else
        retval = 0.0;

return retval;
}







double ind2k(ind,gridsize)

# 	Function converts index value to k-space location. #

int gridsize;
int ind;

{
double k;

k = (double)(ind-gridsize/2) / (double)(gridsize);
return k;
}


double k2ind(k,gridsize)

#	Function calculates the index of the grid point corresponding
	to (k).  Index is fractional - integer index points can
	be calculated from this.	#

int gridsize;
double k;

{
double gridspace;
double index;

gridspace = (double)1.0 / (double)(gridsize);
index = k/gridspace + (double)(gridsize/2);

return index;
}




int k2i(k,gridsize)

#      Function calculates the index of the grid point corresponding
        to (k).  Index is fractional - integer index points can
        be calculated from this.

	Limits to 0:gridsize-1
#

int gridsize;
double k;

{
double gridspace;
int index;

gridspace = (double)1.0 / (double)(gridsize);
index = (int)(k/gridspace + (double)(gridsize/2));
if (index >= gridsize)
	index = gridsize-1;
else
	if (index < 0)
		index = 0;

return index;
}





void getpoint(xlow,xhigh,ylow,yhigh,gridsize,sg_real,sg_imag,
		kxx,kyy,rpart,ipart, gridspace)

#	Function gets the grid point by interpolating. 	#

int xlow;
int xhigh;
int ylow;
int yhigh;
int gridsize;
double *sg_real;
double *sg_imag;
double kxx;
double kyy;
double *rpart;
double *ipart;
double gridspace;

{


*rpart = 0.0;
*ipart = 0.0;

if ((xlow < gridsize) && (xhigh >=0) && (ylow < gridsize) && (yhigh >= 0))
	{
	if (xlow >= 0)
		{
		if (ylow >= 0)
			{
		 	(*rpart) += sg_real[xlow*gridsize+ylow] * convkernal(kxx-ind2k(xlow,gridsize), kyy-ind2k(ylow,gridsize), gridspace);
		 	(*ipart) += sg_imag[xlow*gridsize+ylow] * convkernal(kxx-ind2k(xlow,gridsize), kyy-ind2k(ylow,gridsize), gridspace);
			}
		if (yhigh < gridsize)
			{
		 	(*rpart) += sg_real[xlow*gridsize+yhigh] * convkernal(kxx-ind2k(xlow,gridsize), kyy-ind2k(yhigh,gridsize), gridspace);
		 	(*ipart) += sg_imag[xlow*gridsize+yhigh] * convkernal(kxx-ind2k(xlow,gridsize), kyy-ind2k(yhigh,gridsize), gridspace);
			}
		}

	if (xhigh < gridsize)
		{
		if (ylow >= 0)
			{
		 	(*rpart) += sg_real[xhigh*gridsize+ylow] * convkernal(kxx-ind2k(xhigh,gridsize), kyy-ind2k(ylow,gridsize), gridspace);
		 	(*ipart) += sg_imag[xhigh*gridsize+ylow] * convkernal(kxx-ind2k(xhigh,gridsize), kyy-ind2k(ylow,gridsize), gridspace);
			}
		if (yhigh < gridsize)
			{
		 	(*rpart) += sg_real[xhigh*gridsize+yhigh] * convkernal(kxx-ind2k(xhigh,gridsize), kyy-ind2k(yhigh,gridsize), gridspace);
		 	(*ipart) += sg_imag[xhigh*gridsize+yhigh] * convkernal(kxx-ind2k(xhigh,gridsize), kyy-ind2k(yhigh,gridsize), gridspace);
			}
		}

	}
}




void invgrid(sg_real,sg_imag,gridsize, kx,ky,s_real,s_imag,nsamples)


double *sg_real;
double *sg_imag;
int gridsize;		# Number of points in kx and ky in grid. #
double *kx;
double *ky;
double *s_real;
double *s_imag;
int nsamples;		# Number of samples, total. #



{
int count1, count2;
int gcount1, gcount2;

double gridspace;

double kxind,kyind;

int lowkx;
int lowky;
int highkx;
int highky;

gridspace = (double)(1.0) / (double)(gridsize);


for (count1 = 0; count1 < nsamples; count1++)
	{

	kxind = k2ind(kx[count1],gridsize);
	kyind = k2ind(ky[count1],gridsize);

	lowkx = (int)kxind;
	highkx = (int)(kxind+1.0);
	lowky = (int)kyind;
	highky = (int)(kyind+1.0);


	getpoint(lowkx,highkx,lowky,highky,gridsize,sg_real,sg_imag,
		kx[count1],ky[count1],&(s_real[count1]),&(s_imag[count1]),
		gridspace);

	}

}





void calcdensity(kx,ky,nsamples,dens,gridsize,convwidth)

#	Function calculates the sample density at each point
	using (hopefully) a fast algorithm:

	1)  First the density at each sample point is set to 1.

	2)  For each sample in the set S1, we count through the
	    samples S2 (which are after S1 in the arrays kx and ky).

		If S1 and S2 are within the convolution kernal
		width of each other, the convolution kernal is
		calculated for the vector S1-S2.  This result is
		added to the densities at both S1 and S2.

		I hope that this minimizes the number of times
		that the convolution kernal is calculated!

	Brian Hargreaves - Jan 99.
#

double *kx;	# Kx location of each sample point. #
double *ky;	# Ky location of each sample point. #
int nsamples;	# number of sample points. #
double *dens;		# OUTPUT - Density at each point. #
int gridsize;		# Number of points in kx and ky in grid. #
double convwidth;	# Convolution Kernal width in grid points. #

{
int count1, count2;
double gridspace;

double kxmin,kxmax,kymin,kymax;
double kxx,kyy;
double kx1,ky1;
double kwidth;

double *pdens1, *pdens2, *pkx, *pky, *pkx1, *pky1;
double denschange;

gridspace = (double)(1.0) / (double)(gridsize);
kwidth = gridspace*convwidth;


# ========= Calculate Density Function ======== #

pdens1 = dens;

for (count1 = 0; count1 < nsamples; count1++)
	*(pdens1++) = 1.0;

pdens1 = dens;
pkx1 = kx;
pky1 = ky;

# 	Cycle through each sample point (count1), calculating the
	density contribution from every other point at this point. #

for (count1 = 0; count1 < nsamples; count1++)
        {

	# Find limits so that we don't calculate convolution at
	   more points than we need to. #

        kx1 = *(pkx1++);
        ky1 = *(pky1++);
        kxmin = kx1-kwidth;
        kxmax = kx1+kwidth;
        kymin = ky1-kwidth;
        kymax = ky1+kwidth;

	pkx = pkx1;
	pky = pky1;

	# Get pointer to "remaining points" so that their densities
	   can be modified at the same time, since the effect of
	   density at S1 by S2 is the same as the effect at S2 by S1. #

	pdens2 = pdens1+1;

        for (count2 = count1+1; count2 < nsamples; count2++)
                {
                kxx = *(pkx++);
                kyy = *(pky++);
                if ((kxx>kxmin) && (kxx<kxmax) && (kyy>kymin) && (kyy<kymax))
			{
			denschange = convkernal(kx1-kxx,
                                        ky1-kyy, kwidth);
			*pdens1 += denschange;
			*pdens2 += denschange;
			}
		pdens2++;
                }
	pdens1++;
        }



}


void calcraddensity(kx,ky,nsperrad, nsamples,dens,gridsize,convwidth)

#	Function calculates the sample density at each point
	using (hopefully) a fast algorithm:

	1)  First the density at each sample point is set to 1.

	2)  For each sample in the set S1, we count through the
	    samples S2 (which are after S1 in the arrays kx and ky).

		If S1 and S2 are within the convolution kernal
		width of each other, the convolution kernal is
		calculated for the vector S1-S2.  This result is
		added to the densities at both S1 and S2.

		I hope that this minimizes the number of times
		that the convolution kernal is calculated!

	Brian Hargreaves - Jan 99.
#

double *kx;	# Kx location of each sample point. #
double *ky;	# Ky location of each sample point. #
int nsperrad;	# number of sample points per radial trajectory. #
int nsamples;	# number of sample points. #
double *dens;		# OUTPUT - Density at each point. #
int gridsize;		# Number of points in kx and ky in grid. #
double convwidth;	# Convolution Kernal width in grid points. #

{
int count1, count2;
double gridspace;

double kxmin,kxmax,kymin,kymax;
double kxx,kyy;
double kx1,ky1;
double kwidth;

double *pdens1, *pdens2, *pkx, *pky, *pkx1, *pky1;
double denschange;
int ntheta;

gridspace = (double)(1.0) / (double)(gridsize);
kwidth = gridspace*convwidth;


# ========= Calculate Density Function ======== #

pdens1 = dens;

for (count1 = 0; count1 < nsamples; count1++)
	*(pdens1++) = 1.0;

pdens1 = dens;
pkx1 = kx;
pky1 = ky;

# 	Cycle through each sample point (count1), calculating the
	density contribution from every other point at this point. #

for (count1 = 0; count1 < nsperrad; count1++)
        {

	# Find limits so that we don't calculate convolution at
	   more points than we need to. #

        kx1 = *(pkx1++);
        ky1 = *(pky1++);
        kxmin = kx1-kwidth;
        kxmax = kx1+kwidth;
        kymin = ky1-kwidth;
        kymax = ky1+kwidth;

	pkx = pkx1;
	pky = pky1;

	# Get pointer to "remaining points" so that their densities
	   can be modified at the same time, since the effect of
	   density at S1 by S2 is the same as the effect at S2 by S1. #

	pdens2 = pdens1+1;

        for (count2 = count1+1; count2 < nsamples; count2++)
                {
                kxx = *(pkx++);
                kyy = *(pky++);
                if ((kxx>kxmin) && (kxx<kxmax) && (kyy>kymin) && (kyy<kymax))
			{
			denschange = convkernal(kx1-kxx,
                                        ky1-kyy, kwidth);
			*pdens1 += denschange;
			*pdens2 += denschange;
			}
		pdens2++;
                }
	pdens1++;
        }

#	Now duplicate the densities for other angles... #

ntheta = nsamples/nsperrad;



pdens2 = dens+nsperrad;

for (count1 = 1; count1 < ntheta; count1++)
	{
	pdens1 = dens;
	for (count2 = 0; count2 < nsperrad; count2++)
		*(pdens2++) = *(pdens1++);
	}

}



void gridonly(kx,ky,s_real,s_imag,nsamples, dens,
	sg_real,sg_imag, gridsize, convwidth)

double *kx;
double *ky;
double *s_real;		# Sampled data. #
double *s_imag;
int nsamples;		# Number of samples, total. #

double *dens;		# Output - density function. #

double *sg_real;
double *sg_imag;
int gridsize;		# Number of points in kx and ky in grid. #
double convwidth;


{
int count1, count2;
int gcount1, gcount2;
double gx,gy;
int col;
int loc;
double gridspace;

double kxmin,kxmax,kymin,kymax;
double kxx,kyy;
double kx1,ky1;
double kwidth;
int ixmin,ixmax,iymin,iymax;


gridspace = (double)(1.0) / (double)(gridsize);
kwidth = gridspace*convwidth;

# ========= Zero Output Points ========== #

for (gcount1 = 0; gcount1 < gridsize; gcount1++)
        {
        col = gcount1*gridsize;
        for (gcount2 = 0; gcount2 < gridsize; gcount2++)
                {
                loc = col+gcount2;
                sg_real[loc] = 0.0;
      		sg_imag[loc] = 0.0;
		}
	}

# ========= Find Grid Points ========= #



for (count1 = 0; count1 < nsamples; count1++)
	{
		# Box around k sample location. #

	kxx = kx[count1];
	kyy = ky[count1];
	ixmin = k2i(kxx - kwidth,gridsize);
	ixmax = k2i(kxx + kwidth,gridsize);
	iymin = k2i(kyy - kwidth,gridsize);
	iymax = k2i(kyy + kwidth,gridsize);

	#
	printf("min(%d,%d) - max(%d,%d) \n",ixmin,iymin,ixmax,iymax);
	#

	for (gcount1 = ixmin; gcount1 <= ixmax; gcount1++)
        	{
		gx = (double)(gcount1-gridsize/2) / (double)gridsize;
        	col = gcount1*gridsize;
		for (gcount2 = iymin; gcount2 <= iymax; gcount2++)
			{
                	gy = (double)(gcount2-gridsize/2) / (double)gridsize;
			#
			printf("Separation (%f,%f), c(k)= %f\n",gx-kxx,gy-kyy,
				convkernal(gx-kxx, gy-kyy,kwidth));
			#

			sg_real[col+gcount2] += convkernal(gx-kxx, gy-kyy,
                                kwidth) * s_real[count1]/dens[count1];
			sg_imag[col+gcount2] += convkernal(gx-kxx, gy-kyy,
                                kwidth) * s_imag[count1]/dens[count1];
			}
		}
	}
}