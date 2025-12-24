//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Bias-free algorithm for diffusion-limited aggregation (DLA)
// Yen Lee Loh
// Started 2014-6-3, touched 2014-7-9
//
// To compile:		g++ -O3 dla.cc -o dla
// To run:				./dla
//
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

#include <cmath>    // for exp()
#include <iostream> // for cout
#include <iomanip>  // for setf
#include <fstream>  // for ofstream and ifstream
#include <string>   // for string and stringstream
#include <ctime>    // for clock()
#include <cassert>  // for assert()
#include <algorithm>  // for min() and max()
#include "io.h"     // for readPar(), write_binary(), etc.
#include "math.h"   // for sqr(), rescale(), arithProg(), etc. 
using namespace std;

const int64 RKILLING = (int64)1e14;	// killing radius
const int64 SEED=0;									// SEED=0 means seed RNG using current time
const int64 LMAX=10;								// lmax=17 corresponds to an 65536x65536 grid

const int64 AGGREGATION_PATTERN[][2] 
//= {{0,0},{1,0},{0,1}};   // 2-neighbor aggregation (L)
//= {{0,0},{1,0},{-1,0},{0,-1}};   // 3-neighbor (T)
= {{0,0},{1,0},{-1,0},{0,1},{0,-1}};   // 4-neighbor (+)
//= {{0,0},{1,1},{1,-1},{-1,1},{-1,-1}}; // 4-neighbor (X)
//= {{0,0},{1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}}; // 8-neighbor (*)

enum {FIXED_STEPSIZE,VARIABLE_STEPSIZE};
const int DIFFUSION_STEPSIZE = VARIABLE_STEPSIZE;

enum {SHARP_CIRCLE,FUZZY_ANNULUS};
const int LAUNCHING_METHOD = FUZZY_ANNULUS;



//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// TDiscreteSampler
//
// Sample a discrete probability distribution with N outcomes using Walker's alias method.
//
//	Initialize an object of this class with a probability distribution:
//
//    TDiscrete discreteSampler;
//		discreteSampler.init (  vector<double> ({.1, .4, .2, .3})  );
//
//	Generate an integer 0,1,2,3 from this probability distribution using
//
//    double i = discreteSampler.sample ();
//
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
class TDiscreteSampler {
public:
	int64 n;
	vector<double> pn;	// Probability table (we don't really need to store this)
	vector<double> fn;	// Alias probability table
	vector<int64> an;		// Alias index table
	
	void init (const vector<double> pn) {
		this->pn = pn;
		this->n = pn.size();
		
		//======== Construct Walker alias table =========
		int64 i,j,jmin,jmax;
		double bmin,bmax,s,oon,tol = 1e-11;
		vector<double> bn (n);
		fn.resize(n);
		an.resize(n);
		oon = 1.0/n;
		//--------- Verify that user-supplied pp[]'s sum to unity!
		s=0;
		for (i=0; i<n; i++) {s+=pn[i];}
		cout << setprecision(16);
		cout << n << " probs, total = " << s << endl;
		assert (fabs(s-1) < tol);
		//--------- Set up Walker alias tables
		for (i=0; i<n; i++) {bn[i]=pn[i]-oon; an[i]=i; fn[i]=1.0;}
		for (i=0; i<n; i++) {
			bmin=+1e38; jmin=-1; bmax=-1e38; jmax=-1;
			for (j=0; j<n; j++) {
				if (bmin>bn[j]) {bmin=bn[j]; jmin=j;}
				if (bmax<bn[j]) {bmax=bn[j]; jmax=j;}
			}
			if (bmax-bmin<tol) break;
			an[jmin]=jmax; fn[jmin]=1+bmin*n; bn[jmin]=0; bn[jmax]=bmax+bmin;
		}
	}
	int64 sample () {
		int64 i = irand(n);
		if (drand() > fn[i]) i = an[i]; 
		return i;
	}
};

void testDiscreteSampler () {
	cout << setw(16) << "-------- testDiscreteSampler() --------\n";
	
	TDiscreteSampler discreteSampler;
	
	vector<double> pn;
	pn.push_back(.2);
	pn.push_back(.1);
	pn.push_back(.6);
	pn.push_back(.1);
	int64 n = pn.size();
	discreteSampler.init (pn);
	
	int64 hist[n];
	for (int64 i=0; i<n; ++i) hist[i] = 0;
	int64 itermax = 1000000;
	for (int64 iter=0; iter<itermax; ++iter) {
		int64 i = discreteSampler.sample ();
		hist[i]++;
	}
	
	cout << setprecision(5);
	cout << setw(16) << "pDesired";
	cout << setw(16) << "pSample";
	cout << endl;
	for (int64 i=0; i<n; ++i) {
		cout << setw(16) <<  pn[i];
		cout << setw(16) <<  hist[i]/(double)itermax;
		cout << endl;
	}
}



//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// TWalkToLine
//
//		Suppose a particle starts at position (0,r)
//		and executes a random walk until it touches the line y=0.
//
//		Fxy(x,y)    returns the resistance Green function at point64 (x,y)
//		pxy(x,y)		returns the probability of touching the line at point64 (x,0)
//		xSample(y)	returns a value for x sampled from the above probability distribution
//
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
class TWalkToLine {
public:
	double FxyExact[64][64];
	
	TWalkToLine () {
		cout << "-------- TWalkToLine() --------\n";
		cout << "Loading SquareLatticeGreenFunction64.raw (F_{xy} in paper) \n";
		ifstream ifs ("SquareLatticeGreenFunction64.raw");
		ifs.read ((char*)FxyExact, 64*64*sizeof(double));
		ifs.close();	
	}
	//-------- Return the resistance Green function F(x,y)
	// Note that this could be optimized further.
	double FxySeries (int64 x, int64 y) {
		const double eulerGamma = 0.5772156649015329;
		const double eulerGammaPlusSqrt8 = 1.6169364357414509;
		double rsq = (double)x*x + (double)y*y;
		double oor2 = 1./rsq;
		double oor4 = sqr(oor2);
		double oor6 = oor2*oor4;
		double oor8 = sqr(oor4);
		double phi = atan((double)y / x);
		double cos4phi = cos (4*phi);
		double cos8phi = cos (8*phi);
		double cos12phi = cos (12*phi);
		double cos16phi = cos (16*phi);
		return 1.0/(2*M_PI) 
		* (
			 eulerGammaPlusSqrt8 + 0.5*log(rsq)
			 - (
					+ oor2 * (1/12.*cos4phi)
					+ oor4 * (3/40.*cos4phi + 5/48.*cos8phi)
					+ oor6 * (51/112.*cos8phi + 35/72.*cos12phi)
					+ oor8 * (217/320.*cos8phi + 45/8.*cos12phi + 1925/384.*cos16phi)
					)
			 );
	}
	double Fxy (int64 x, int64 y) {
		x = abs(x);
		y = abs(y);
		if (x<=60 && y<=60) return FxyExact[x][y];
		else                return FxySeries (x, y);
	}
	
	//-------- Target distribution
	double pxy (int64 x, int64 h) {
		return Fxy(x,1+h) - Fxy(x,1-h);
	}
	//-------- Envelope distribution
	double qxy (int64 x, int64 h) {
		return (atan ((x+0.5) / h) - atan ((x-0.5) / h)) / M_PI;
	}
	int64 xSample (int64 h) {
		double p0 = pxy (0, h);
		double q0 = qxy (0, h);
		double c = q0 / p0;
		int64 x;
		for (int64 iter=0; ; ++iter) {
			double xRaw = h * tan ((drand()-0.5)*M_PI );
			if			(xRaw > +RKILLING)	x = int64max; // indicate overflow
			else if (xRaw < -RKILLING)	x = int64min;
			else											x = (int64) round(xRaw);
			int64 xAbs = abs (x);
			double p = pxy (xAbs, h);
			double q = qxy (xAbs, h);
			double pAccept = c * p / q;
			if (drand() < pAccept) break;
			
			if (iter>100) {
				cerr << "Warning: retrial sampling has taken " << iter << " iterations!" << endl;
			}
		} 
		return x;
	}
	
} walkToLine;

void testWalkToLine () {
	cout << setw(16) << "-------- testWalkToLine() --------\n";
	int64 xmin = -10;
	int64 xmax = +10;
	int64 buffer[xmax-xmin], *hist = &buffer[0-xmin];
	for (int64 x=xmin; x<xmax; ++x) 	hist[x] = 0;
	
	int64 y = 8;
	int64 imax = 9000000;
	for (int64 i=0; i<imax; ++i) {
		int64 x = walkToLine.xSample (y);
		if (x>=xmin && x<xmax) 	hist[x]++;
	}	
	cout << fixed << setprecision(4);	
	cout << setw(16) << "x";
	cout << setw(16) << "pTrue";
	cout << setw(16) << "pSample";
	cout << endl;
	for (int64 x=xmin; x<xmax; ++x) {
		cout << setw(16) << x;
		cout << setw(16) << walkToLine.pxy(x, y);
		cout << setw(16) << hist[x] / (double)imax;
		cout << endl;
	}
}

double FFxy (int64 x, int64 y) {return walkToLine.Fxy (x, y);} // For testing purposes only!!!
void testGreenFunction () {
	double rhs = 0.;
	double residual = 0.;
	for (int64 x=0; x<100; ++x) {
		for (int64 y=0; y<100; ++y) {
			rhs = 4*FFxy(x,y) - FFxy(x+1,y) - FFxy(x-1,y) - FFxy(x,y+1) - FFxy(x,y-1);
			if (x==0 && y==0) rhs += 1.;
			residual = max(residual, fabs(rhs));
		}	
	}
	cout << "Maximum residual in Poisson equation over a 100x100 grid = " << residual << endl;
}




//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// TWalkOutToSquare
// 
//		Suppose a particle starts at position (0,0)
//		and executes a random walk until it touches the square with corners
//    (-r,-r) and (r,r).
//
//		pxr(x,r)		returns 4 times the probability of touching the square at point64 (x,r)
//		xSample(r)	returns a value for x sampled from the probability distribution pxr(x,r)
//		findxySample(r,x,y)		sets (x,y) to a random location on the square 
//													according to the appropriate probability distribution.
//
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
const int64 levels = 9;
class TWalkOutToSquare {
public:
	TDiscreteSampler discreteSampler[levels];
	
	TWalkOutToSquare () {
		cout << "-------- TWalkOutToSquare() --------\n";
		cout << "Loading DirichletGFs.raw (4*Psq_{lx} in paper) \n";
		ifstream ifs ("DirichletGFs.raw");
		for (int64 l=0; l<levels; ++l) {
			int64 rsquare = (1 << l);
			int64 nvalues = rsquare*2 - 1;
			
			vector<double> pn (nvalues);
			ifs.read ((char*)pn.data(), nvalues*sizeof(double));
			
			discreteSampler[l].init (pn);
		}
		ifs.close();	
	}
	double pxr (int64 x, int64 r) { // This is not actually required for the algorithm
		int64 l;
		switch (r) {	// Determine which discreteSampler object to use
			case 1: l=0; break;
			case 2: l=1; break;
			case 4: l=2; break;
			case 8: l=3; break;
			case 16: l=4; break;
			case 32: l=5; break;
			case 64: l=6; break;
			case 128: l=7; break;
			case 256: l=8; break;
			default: l=-1; break;
		}
		if (l==-1) {cerr << r; abort ();}
		return discreteSampler[l].pn[r - 1 + x];
	}
	int64 xSample (int64 r) {	
		int64 l;
		switch (r) {
			case 1: l=0; break;
			case 2: l=1; break;
			case 4: l=2; break;
			case 8: l=3; break;
			case 16: l=4; break;
			case 32: l=5; break;
			case 64: l=6; break;
			case 128: l=7; break;
			case 256: l=8; break;
			default: l=-1; break;
		}
		if (l==-1) {cerr << r; abort ();}
		return discreteSampler[l].sample () - (r-1);
	}
	void findxySample (int64 rSquare, int64 &xLaunch, int64 &yLaunch) {	
		int64 x = xSample (rSquare);
		int64 y = rSquare;
		switch (irand(4)) {
			case 0: xLaunch=x; yLaunch=y; break;
			case 1: xLaunch=x; yLaunch=-y; break;
			case 2: xLaunch=y; yLaunch=x; break;
			case 3: xLaunch=-y; yLaunch=x; break;
		}
		//---- WARNING: When compiling with -O3, sometimes the switch statement is optimized away
		// and x and y do not get set.
	}
	
} walkOutToSquare;

void testWalkOutToSquare () {
	cout << setw(16) << "-------- testWalkOutToSquare() --------\n";
	int64 r = 8;
	int64 xmin = -r;
	int64 xmax = +r+1;
	int64 nx = xmax-xmin;
	int64 buffer[nx], *hist = &buffer[0-xmin];
	for (int64 x=xmin; x<xmax; ++x) 	hist[x] = 0;
	int64 imax = 9000000;
	for (int64 i=0; i<imax; ++i) {
		int64 x = walkOutToSquare.xSample (r);
		if (x>=xmin && x<xmax) 	hist[x]++;
	}	
	cout << fixed << setprecision(4);	
	cout << setw(16) << "x";
	cout << setw(16) << "pTrue";
	cout << setw(16) << "pSample";
	cout << endl;
	for (int64 x=xmin; x<xmax; ++x) {
		cout << setw(16) << x;
		cout << setw(16) << walkOutToSquare.pxr(x,r);
		cout << setw(16) << hist[x] / (double)imax;
		cout << endl;
	}
}








//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// FUZZY LAUNCHING ANNULUS 
// Launch the particle as follows.
// Pick the radius from a Kaiser-Bessel window function distribution on (rInner,rOuter).
// Pick the angle from a uniform distribution on [0,2pi].
// This reduces launching bias to below 10^(-12).
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
const double betaKaiser = 24.;
const double cKaiser = 9.060322906269867e-10;
const double sigmaGauss = 0.20630813344176394; // this is for betaKaiser=24.
const double cGauss = 1.9648389200167407;

double fKaiser (double x) {
	return cKaiser * besselI0 ( betaKaiser * sqrt(1 - x*x) );
}
double fEnvelope (double x) {
	return cGauss * exp (- sqr(x) * (0.5/sqr(sigmaGauss)));
}
double drandKaiser () {
	double x,y;
	while (true) {
		x = sigmaGauss * drandNormal();
		if (fabs(x) < 1 && drand() < fKaiser(x) / fEnvelope(x)) break;
	}
	return x * (2*irand(2) - 1);
}
void pickLaunchPointFromAnnulus (int64 &xRel, int64 &yRel, double rInner, double rOuter) {	
	double r = 0.5*(rInner+rOuter) + 0.5*(rOuter-rInner)*drandKaiser();
	double phi = drand()*2*M_PI;
	xRel = (int64) round( r * cos(phi) );
	yRel = (int64) round( r * sin(phi) );
}
void testDrandKaiser () {
	int64 x,y;
	ofstream ofs ("testDrandKaiser.dat");
	ofs << setprecision(16);
	for (int i=0; i<1000000; ++i) {
		ofs << setw(16)  << drandKaiser() << endl;
	}
	ofs.close ();
}
void testLaunchParticle () {
	int64 x,y;
	ofstream ofs ("testLaunchParticle.dat");
	for (int i=0; i<1000000; ++i) {
		pickLaunchPointFromAnnulus (x, y, 200., 400.);
		ofs << setw(5) << x << setw(5) << y << endl;
	}
	ofs.close ();
}











//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// DLA data structures and algorithms
//
//
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
const int64 lmax = LMAX; // lmax=14 corresponds to an 8192x8192 grid
const int64 xmax = (1 << (lmax-1));
const int64 ymax = (1 << (lmax-1));
const int64 xCen = xmax/2;
const int64 yCen = ymax/2;
typedef unsigned char byte;
byte **ss; // this is a bit array!
char sDummy;
int64 xminBound=int64max,xmaxBound=int64min;
int64 yminBound=int64max,ymaxBound=int64min; //bounding box of cluster and halo
int64 rBound=int64min;

int64 mTotal = 0;	// cluster mass
double xTotal = 0.;
double yTotal = 0.;
double xsqTotal = 0.;
double ysqTotal = 0.;
double rGyration = 0.;

const int nAngles = 24; // for gathering radius-vs-angle stats
double radVsAngle[nAngles];

//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Utilities for reading the aggregation pattern.  
// For example, the + aggregation pattern is defined by an array of 5 points
// representing the walker's position and its EWNS neighbors.
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
const int64 nAggregation = sizeof(AGGREGATION_PATTERN)/(2*sizeof(int64));
int64 xAggregation (int64 i) {return AGGREGATION_PATTERN[i][0]; }
int64 yAggregation (int64 i) {return AGGREGATION_PATTERN[i][1]; }

//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// The configuration of particles is described by slxy(lmax-1,x,y).
// We also store coarse-grained configurations as slxy([0,1,2,..,lmax-2],x,y).
// The raw data is stored in the buffer array ss.
//		setslxy(l,x,y)			sets the state at level l and position (x,y)
//		getslxy(l,x,y)			returns the state
// Note that bit 0 is the MSB, and bit 7 is the LSB.
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
void clearslxy (int64 l, int64 x, int64 y) {
	int64 xmaxl = (1 << l);
	int64 ymaxl = (1 << l);
	if (x<0 || x>=xmaxl || y<0 || y>=ymaxl) return; // bounds check
	int64 ixy = x*ymaxl + y;
	int64 iword = ixy >> 3;				// iword = ixy/8;
	int64 ibit = ixy & 7;					// ibit = ixy%8;
	ss[l][iword] &= ~(byte(128) >> ibit);	// clear bit using AND NOT
}
void setslxy (int64 l, int64 x, int64 y) {
	int64 xmaxl = (1 << l);
	int64 ymaxl = (1 << l);
	if (x<0 || x>=xmaxl || y<0 || y>=ymaxl) return; // bounds check
	int64 ixy = x*ymaxl + y;
	int64 iword = ixy >> 3;				// iword = ixy/8;
	int64 ibit = ixy & 7;					// ibit = ixy%8;
	ss[l][iword] |= (byte(128) >> ibit);	// set bit using OR
}
byte getslxy (int64 l, int64 x, int64 y) {
	int64 xmaxl = (1 << l);
	int64 ymaxl = (1 << l);
	if (x<0 || x>=xmaxl || y<0 || y>=ymaxl) return 0; // bounds check
	int64 ixy = x*ymaxl + y;
	int64 iword = ixy >> 3;							// iword = ixy/8;
	int64 ibit = ixy & 7;								// ibit = ixy%8;
	return (ss[l][iword] >> (7-ibit)) & byte(1);	// read bit using AND
}
byte getsxy (int64 x, int64 y) {
	return getslxy (lmax-1, x, y);
}
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// initHierarchicalArray(): set up the ss array
// Also set up some stats
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
void initHierarchicalArray () {
	ss = new byte*[lmax];
	for (int64 l=0; l<lmax; ++l) {
		int64	xmaxl = (1 << l);
		int64	ymaxl = (1 << l);
		ss[l] = new byte[(xmaxl*ymaxl+7)/8]; // bit array!
		for	(int64 x=0; x<xmaxl; ++x) {
			for (int64 y=0; y<ymaxl; ++y) {
				clearslxy(l, x, y);
			}
		}		
	}
	
	for (int n=0; n<nAngles; ++n) {
		radVsAngle[n] = 0.;
	}
}

//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Mark site (x,y) as a sticky site at all levels of the hierarchy
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
void mark (int64 x, int64 y) {
	for (int64 l=lmax-1; l>=0; --l) {			
		setslxy (l, x, y);
		x >>= 1;
		y >>= 1;
	}	
}
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Freeze a walker at position (x,y).
// For (x,y), and for each of its neighbors according to the aggregation rule,
// mark them as sticky sites, and do this every level of the hierarchy!
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
void freeze (int64 x, int64 y) {
	//-------- Update cluster mass, radius of gyration, max radius at each angle, etc.
	mTotal += 1;
	xTotal += x;
	yTotal += y;
	xsqTotal += sqr(x);
	ysqTotal += sqr(y);
	double xAverage = xTotal / mTotal;
	double yAverage = yTotal / mTotal;
	double xsqAverage = xsqTotal / mTotal;
	double ysqAverage = ysqTotal / mTotal;
	double rsqGyration = xsqAverage - sqr(xAverage) + ysqAverage - sqr(yAverage);	
	rGyration = sqrt (rsqGyration);
	
	double r = hypot (y-yCen+0., x-xCen+0.);
	double phi = atan2 (y-yCen, x-xCen);
	int nAngle = modulo ( (int)round (phi/(2*M_PI)*nAngles), nAngles);
	radVsAngle[nAngle] = max (	radVsAngle[nAngle], r);
	
	//-------- Update bounding radius
	rBound = max (rBound, (int64)hypot(x - 1.0*xCen, y - 1.0*yCen));
	
	//-------- Update bounding box
	// ASSUME THAT AGGREGATION PATTERN IS CONTAINED WITHIN +/- 2 SITES.
	// NEAREST-NEIGHBOR AGGREGATION WOULD ONLY USE +/- 1 SITE.
	xminBound = min (xminBound, x-2);
	yminBound = min (yminBound, y-2);
	xmaxBound = max (xmaxBound, x+2);
	ymaxBound = max (ymaxBound, y+2);
	
	//-------- Mark walker's site and new sticky sites
	mark (x, y);
	for (int64 i=0; i<nAggregation; ++i) {
		mark (x+xAggregation(i), y+yAggregation(i));
	}
}
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Initialize the seed cluster
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
void initSeedCluster () {
	//======== Seed cluster of shape .
	freeze (xCen, yCen);	
	//======== Seed cluster of shape :
	//	freeze (xCen, yCen);	freeze (xCen+12, yCen);	
	//	freeze (xCen, yCen+12);	freeze (xCen+12, yCen+12);	
	//======== Seed cluster of shape -
	//	for (int x=xCen-100; x<xCen+100; ++x)	freeze (x, yCen);	
	//======== Seed cluster of shape /
	//for (int x=xCen-100; x<xCen+100; ++x) freeze (x, x);	
	//======== Seed cluster of shape +
	//for (int i=-1000; i<=1000; ++i) {freeze (xCen+i, yCen);	freeze (xCen, yCen+i);}	
	//======== Seed cluster of shape X
	//for (int i=-100; i<=100; ++i) {freeze (xCen+i, yCen+i);	freeze (xCen-i, yCen+i);}	
}
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Launch a new particle using circle and snap-to-grid
// or using Kaiser-Bessel fuzzy annulus
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
void launchParticle (int64 &xCur, int64 &yCur) {	
	if (LAUNCHING_METHOD==SHARP_CIRCLE) {
		double r = ::rBound + 2.;
		double phi = drand()*2*M_PI;
		int64 xRel,yRel;
		xRel = (int64) round( r * cos(phi) );
		yRel = (int64) round( r * sin(phi) );	
		xCur = ::xCen + xRel;
		yCur = ::yCen + yRel;		
	}
	else if (LAUNCHING_METHOD==FUZZY_ANNULUS) {
		double rInner = 2.*max (::rBound, int64(20));
		double rOuter = 4.*max (::rBound, int64(20));
		int64 xRel,yRel;
		pickLaunchPointFromAnnulus (xRel, yRel, rInner, rOuter);
		xCur = ::xCen + xRel;
		yCur = ::yCen + yRel;
	}
}
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Diffuse a particle using fixed or variable stepsize methods
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
void diffuseFixedStepsize (int64 &xCur, int64 &yCur) {		
	//%%%%%%%% Take a step by one lattice spacing in E,W,N, or S direction
	int64 xChange,yChange;
	walkOutToSquare.findxySample (1, xChange, yChange);		
	xCur += xChange;
	yCur += yChange;		
}
void diffuseVariableStepsize (int64 &xCur, int64 &yCur) {	
	//%%%%%%%% Particle is within bounding box of cluster
	// Start at l=2 or l=lmax-9 (256x256 blocks), whichever is larger.
	// Keep going to finer levels of hierarchy 
	// until there are no sticky sites in neighborhood,
	// or until we get down to the finest level.
	int64 lBlock,blockSize,xBlock,yBlock;
	for (lBlock=max((int64)2,lmax-9); lBlock<=lmax-1; ++lBlock) {
		//======== Examine level lBlock of the hierarchical array
		blockSize = (1 << (lmax-lBlock-1));
		xBlock = (xCur >> (lmax-lBlock-1));
		yBlock = (yCur >> (lmax-lBlock-1));					
		//======== Examine a 3x3 region of blocks surrounding the current block
		bool neighborExists = false;
		for (int64 u=xBlock-1; u<=xBlock+1; ++u) {
			for (int64 v=yBlock-1; v<=yBlock+1; ++v) {
				if (getslxy(lBlock, u, v) > 0) {
					neighborExists = true;
					break;
				}
			}
		}
		//======== Zoom in until we reach a scale where the neighborhood is empty
		if (!neighborExists) break;	
	}
	//%%%%%%%% Walk out to a square of side 2*blockSize
	int64 xChange,yChange;
	walkOutToSquare.findxySample (blockSize, xChange, yChange);		
	xCur += xChange;
	yCur += yChange;								
}
void diffuse (int64 &xCur, int64 &yCur) {	
	if (DIFFUSION_STEPSIZE==FIXED_STEPSIZE) 
		diffuseFixedStepsize (xCur, yCur);
	else if (DIFFUSION_STEPSIZE==VARIABLE_STEPSIZE) 
		diffuseVariableStepsize (xCur, yCur);
}
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Return the particle towards the bounding box of the cluster
// using accelerated diffusion using the walk-to-line algorithm
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
void returnParticleTowardsBoundingBox (int64 &xCur, int64 &yCur) {	
	int64 xTarget,xDist,yTarget,yDist;
	if      (xCur < xminBound) {xTarget=xminBound; xDist=xminBound-xCur;}
	else if (xCur > xmaxBound) {xTarget=xmaxBound; xDist=xCur-xmaxBound;}
	else                       {xDist=0; }
	if      (yCur < yminBound) {yTarget=yminBound; yDist=yminBound-yCur;}
	else if (yCur > ymaxBound) {yTarget=ymaxBound; yDist=yCur-ymaxBound;}
	else                       {yDist=0; }
	
	bool overflowed = false;	
	if (yDist > xDist) {
		//-------- Diffuse a distance yDist onto the horizontal line y=yTarget
		yCur = yTarget;
		int64 xChange = walkToLine.xSample (yDist);
		if (xChange>=int64max || xChange<=int64min) overflowed=true;
		xCur += xChange;
	}
	else {
		//-------- Diffuse a distance xDist onto the vertical line x=xTarget
		xCur = xTarget;
		int64 yChange = walkToLine.xSample (xDist);
		if (yChange>=int64max || yChange<=int64min) overflowed=true;
		yCur += yChange;
	}				
	if (overflowed) {
		launchParticle (xCur, yCur);
	}
}



//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Write the bit array describing the current cluster configuration
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
void writeConfig (const char *filename, int64 l) {
	int64 xmaxl=(1<<l);
	int64 ymaxl=(1<<l);
	ofstream ofs (filename);
	ofs.write ((char*)ss[l], (xmaxl*ymaxl+7)/8); // Dump the bitarray
	ofs.close ();
}







//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
//                          MAIN
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
int main (int argc, char** argv) {
	ofstream ofsMassRadius ("mr.dat");
	ofstream ofsAngDep ("angdep.dat");
	ofstream ofsPars ("pars.dat");
	ofsPars << LMAX << endl; // this is useful for coarsen.cc
	ofsPars.close ();
	
	if (SEED==0)	srandom (time(NULL)+clock());
	else					srandom (SEED);
	
	clock_t clockStart;
	time_t timeStart;
	timeStart = time(NULL);
	clockStart = clock();
	
	//======== Test key statistical subroutines
	//	testGreenFunction ();
	//	testDiscreteSampler ();
	//	testWalkToLine ();
	//	testWalkOutToSquare ();
	//	testBesselI0 ();
	//	testDrandKaiser ();
	//	testLaunchParticle ();
	//	exit(1);
	
	initHierarchicalArray ();		
	
	initSeedCluster ();
	
	cout << "================== Starting DLA simulation ================\n";
	cerr << setw(10) << "Mass";
	cerr << setw(15) << "Walltime (ms)";
	cerr << setw(10) << "Rbounding";
	cerr << setw(10) << "Rgyration";
	cerr << endl;

	while (true) {
		
		int64 xCur,yCur;
		launchParticle (xCur, yCur);
		
		while (true) {
			
			if (xminBound<=xCur && xCur<=xmaxBound && yminBound<=yCur && yCur<=ymaxBound) {		
				//-------- Particle is within bounding box; use walk-out-to-square diffusion
				diffuse (xCur, yCur);				
			}
			else {
				//-------- Particle is outside bounding box; use walk-to-line diffusion
				returnParticleTowardsBoundingBox (xCur, yCur);
			}
			if (getsxy(xCur, yCur) > 0) {
				//-------- Particle is at sticky site
				freeze (xCur, yCur);
				break;
			}										
		}
		
		//%%%%%%%% Output stats
		if (isPowerOfTwo(mTotal) || (mTotal>262144 && mTotal%262144==0)) {
			cerr << setw(10) << mTotal;
			cerr << setw(15) << (int64) (  (clock()-clockStart+0.)/CLOCKS_PER_SEC*1000 );
			cerr << setw(10) << rBound;
			cerr << setw(10) << rGyration;
			cerr << endl;
			ofsMassRadius << mTotal << " ";
			ofsMassRadius << rBound << " ";
			ofsMassRadius << rGyration << " ";
			ofsMassRadius << endl;
			for (int n=0; n<nAngles; ++n) {
				ofsAngDep << radVsAngle[n] << " ";
			}
			ofsAngDep << endl;			
		}		
		
		//%%%%%%%% If cluster is touching boundary of system, we must stop
		if (xminBound<=2 || xmaxBound>=xmax-2 ||
				yminBound<=2 || ymaxBound>=ymax-2 )  break;
	}
	
		
	cout << endl;
	writeConfig ("sfinal.raw", lmax-1);
	cout << "================== Finished ================\n";	
}