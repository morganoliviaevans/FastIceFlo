#include <cstdio>
#include <iostream>
#include <string.h>
#include <cmath>
#include <nvtx3/nvToolsExt.h>
using namespace std;

/* Define GPU specific variables */
#define GPU_ID   0
#define BLOCK_Xe 128 
#define BLOCK_Xv 128

/* Device norm subroutine */
#define blockId       (blockIdx.x)
#define threadId      (threadIdx.x)
#define isBlockMaster (threadIdx.x==0)

#include "helpers.h"

#define div10 0.1
#define div30 0.0333333333
#define div60 0.0166666667
  
/*------- PT1 Inputs and Outputs -------*/
/*
Inputs: ftype* vx
        ftype* vy 
        ftype* alpha 
        ftype* beta
        int* index
        ftype* kvx
        ftype* kvy
        ftype* etan
        ftype* Helem
        ftype* areas
        bool* isice
        ftype* Eta_nbe
        ftype* rheology_B
        ftype n_glen
        ftype eta_0
        ftype rele
        int nbe

Outputs: ftype* kvx
         ftype* kvy
         ftype* etan
         ftype* Eta_nbe
*/

void PT1(ftype* vx, ftype* vy, ftype* alpha, ftype* beta, int* index,  ftype* kvx, ftype* kvy, ftype* etan,  ftype* Helem, ftype* areas, bool* isice, ftype* Eta_nbe, ftype* rheology_B, ftype n_glen, ftype eta_0, ftype rele,int nbe){ 
    
    // The OpenACC copyin() clause is used to specify data that needs to be copied from the CPU to the GPU before kernel execution
    #pragma acc data copyin(isice[nbe], rheology_B[nbe], etan[nbe], Helem[nbe], areas[nbe], kvx[nbe], kvy[nbe], Eta_nbe[nbe])
    {
        /* The OpenACC #pragma acc parallel loop directive instructs the compiler to parallelize the loop immediately following it, async(1) = cudaStreamCreate(&stream1); */
        #pragma acc parallel loop async(1)
        for (int ix = 0; ix < nbe; ix += 1) { 
            
            if (isice[ix]) {

                ftype Localalpha[3];
                ftype Localbeta[3];
                ftype Localvx[3];
                ftype Localvy[3];
            
                for (int i = 0; i < 3; i++) {
                Localalpha[i] = alpha[ix*3+i];
                Localbeta[i]  = beta[ix*3+i];
                Localvx[i]    = vx[index[ix*3+i]-1];
                Localvy[i]    = vy[index[ix*3+i]-1];
                }
            
                ftype dvxdx = Localvx[0]*Localalpha[0] + Localvx[1]*Localalpha[1] + Localvx[2]*Localalpha[2];
                ftype dvxdy = Localvx[0]* Localbeta[0] + Localvx[1]* Localbeta[1] + Localvx[2]* Localbeta[2];
                ftype dvydx = Localvy[0]*Localalpha[0] + Localvy[1]*Localalpha[1] + Localvy[2]*Localalpha[2];
                ftype dvydy = Localvy[0]* Localbeta[0] + Localvy[1]* Localbeta[1] + Localvy[2]* Localbeta[2];

                ftype  eps_xx = dvxdx;
                ftype  eps_yy = dvydy;
                ftype  eps_xy = .5*(dvxdy+dvydx);
                ftype  EII2   = eps_xx*eps_xx + eps_yy*eps_yy + eps_xy*eps_xy + eps_xx*eps_yy;
                ftype  eta_it = 5e+13;

            if (EII2>0.) eta_it = rheology_B[ix]/(2*__powf(EII2,(n_glen-1.)/(2*n_glen)));             //------------------------------ ** FIX ME **
            
                etan[ix] = min(__expf(rele*__logf(eta_it) + (1.0-rele)*__logf(etan[ix])),eta_0*1e5);  //------------------------------ ** FIX ME **

                /* Viscous Deformation */
                ftype tmp_2hele_etan_areas = 2 * Helem[ix] * etan[ix] * areas[ix];

                for (int i = 0; i < 3; i++) {
                    kvx[ix*3+i] = tmp_2hele_etan_areas * ((2 * eps_xx + eps_yy) * Localalpha[i] + eps_xy * Localbeta[i]);
                    kvy[ix*3+i] = tmp_2hele_etan_areas * ((2 * eps_yy + eps_xx) * Localbeta[i]  + eps_xy * Localalpha[i]);            
                }
            }
            Eta_nbe[ix] = etan[ix] * areas[ix];
        }
    }
}

/*------- PT2_x Inputs and Outputs -------*/
/* 
Inputs: ftype* kvx
        ftype* groundedratio
        ftype* areas
        int* index
        ftype* alpha2
        ftype* vx
        ftype* gr_a_alpha2
        bool* isice
        int nbe

Outputs: ftype* kvx
*/

void PT2_x(ftype* kvx, ftype* groundedratio, ftype* areas, int* index, ftype* alpha2, ftype* vx, ftype* gr_a_alpha2, bool* isice, int nbe) {

    // The OpenACC copyin() clause is used to specify data that needs to be copied from the CPU to the GPU before kernel execution
    #pragma acc data copyin(isice[nbe], kvx[nbe], groundedratio[nbe], gr_a_alpha2[nbe], vx[nbe], index[nbe])
    {
        /* The OpenACC #pragma acc parallel loop directive instructs the compiler to parallelize the loop immediately following it */
        #pragma acc parallel loop async(1)
        for (int ix = 0; ix < nbe; ix += 1) { 
            /*Add basal friction*/
            if (groundedratio[ix] > 0.) {

                int n3 = ix * 3;

                ftype myLocalIndex[3][3];

                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        int j_index = index[n3 + j] - 1;
                        myLocalIndex[i][j] = gr_a_alpha2[n3 + i] * vx[j_index];
                    }
                }

                ftype tempOutput[3];

                for (int k = 0; k < 3; k++) {
                    tempOutput[k] = kvx[n3 + k];
                }   

                ftype division;

                for (int k = 0; k < 3; k++) {
                    for (int i = 0; i < 3; i++) {
                        for (int j = 0; j < 3; j++) {
                            ftype temp = myLocalIndex[i][j];

                            if (i == j && j == k) { division = div10; } 
                            else if ((i!=j) && (j!=k) && (k!=i)) { division = div60; } 
                            else { division = div30; }

                            tempOutput[k] = isice[ix] * tempOutput[k] + temp*division;
                        }
                    }
                } 
                for (int k = 0; k < 3; k++) {
                    kvx[n3 + k] = tempOutput[k];
                }  
            }
        }
    }
}

/*------- PT2_y Inputs and Outputs -------*/
/* 
Inputs: ftype* kvy
        ftype* groundedratio
        ftype* areas
        int* index
        ftype* alpha2
        ftype* vy
        ftype* gr_a_alpha2
        bool* isice
        int nbe

Outputs: ftype* kvy
*/

void PT2_y(ftype* kvy, ftype* groundedratio, ftype* areas, int* index, ftype* alpha2, ftype* vy, ftype* gr_a_alpha2, bool* isice, int nbe) {

    // The OpenACC copyin() clause is used to specify data that needs to be copied from the CPU to the GPU before kernel execution
    #pragma acc data copyin(isice[nbe], kvy[nbe], groundedratio[nbe], gr_a_alpha2[nbe], vy[nbe], index[nbe])
    {
        /* The OpenACC #pragma acc parallel loop directive instructs the compiler to parallelize the loop immediately following it */
        #pragma acc parallel loop async(2)
        for (int ix = 0; ix < nbe; ix += 1) { 
            /*Add basal friction*/
            if (groundedratio[ix] > 0.) {

                int n3 = ix * 3;

                ftype myLocalIndex[3][3];

                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        int j_index = index[n3 + j] - 1;
                        myLocalIndex[i][j] = gr_a_alpha2[n3 + i] * vy[j_index];
                    }
                }

                ftype tempOutput[3];

                for (int k = 0; k < 3; k++) {
                    tempOutput[k] = kvy[n3 + k];
                }

                ftype division;

                for (int k = 0; k < 3; k++) {
                    for (int i = 0; i < 3; i++) {
                        for (int j = 0; j < 3; j++) {

                            ftype temp = myLocalIndex[i][j];

                            if (i == j && j == k) { division = div10; } 
                            else if ((i!=j) && (j!=k) && (k!=i)) { division = div60; } 
                            else { division = div30; }

                            tempOutput[k] = isice[ix] * tempOutput[k] + temp*division;
                        }
                    }
                } 
                for (int k = 0; k < 3; k++) { kvy[n3 + k] = tempOutput[k]; } 
            }
        }
    }
}

/*------- PT3_a Inputs and Outputs -------*/
/* 
Inputs: ftype* kvx
        ftype* kvy
        ftype* Eta_nbe
        ftype* eta_nbv
        int* connectivity
        int* columns
        ftype* KVx
        ftype* KVy
        int nbv

Outputs: ftype* KVx
         ftype* KVy
         ftype* eta_nbv
*/

void PT3_a(ftype* kvx, ftype* kvy, ftype* Eta_nbe, ftype* eta_nbv, int* connectivity, int* columns, ftype* KVx, ftype* KVy, int nbv) {
	
    // The OpenACC copyin() clause is used to specify data that needs to be copied from the CPU to the GPU before kernel execution
    #pragma acc data copyin(KVx[nbv], KVy[nbv], kvx[nbv], kvy[nbv], eta_nbv[nbv], Eta_nbe[nbv], columns[nbv], connectivity[nbv])
    {
        /* The OpenACC #pragma acc parallel loop directive instructs the compiler to parallelize the loop immediately following it */
        #pragma acc parallel loop async(1)
        for (int ix = 0; ix < nbv; ix += 1) { 

            KVx[ix] = 0.;
            KVy[ix] = 0.;
            int localColumns[8];
            int localConnectivity[8];

            for (int j = 0; j < 8; j++) {
                localConnectivity[j] = connectivity[(ix * 8  + j)];
                localColumns[j] = columns[(ix * 8  + j)];
            } 

            for (int j = 0; j < 8; j++) {
                if (localConnectivity[j] != 0) {
                    KVx[ix] = KVx[ix] + kvx[((localConnectivity[j])-1) *3 + ((localColumns[(j)]))];
                    KVy[ix] = KVy[ix] + kvy[((localConnectivity[j])-1) *3 + ((localColumns[(j)]))];
                    eta_nbv[ix] = eta_nbv[ix] + Eta_nbe[localConnectivity[j]-1];
                }
                else { break; }
            }
        }
    }
} 

/*------- PT3_b Inputs and Outputs -------*/
/* 
Inputs: ftype* KVx
        ftype* KVy
        ftype* Fvx
        ftype* Fvy
        ftype* dVxdt
        ftype* dVydt
        ftype* rho_ML
        ftype damp
        int nbv

Outputs: ftype* dVxdt
         ftype* dVydt
*/ 

void PT3_b(ftype* KVx, ftype* KVy, ftype* Fvx, ftype* Fvy, ftype* dVxdt, ftype* dVydt, ftype* rho_ML, ftype damp, int nbv) {

    ftype ResVx; 
    ftype ResVy;  

    // The OpenACC copyin() clause is used to specify data that needs to be copied from the CPU to the GPU before kernel execution
    #pragma acc data copyin(KVx[nbv], KVy[nbv], Fvx[nbv], Fvy[nbv], rho_ML[nbv], dVxdt[nbv], dVydt[nbv])
    {
        /* The OpenACC #pragma acc parallel loop directive instructs the compiler to parallelize the loop immediately following it */
        #pragma acc parallel loop async(1)
        for (int ix = 0; ix < nbv; ix += 1) {
            /* 1. Get time derivative based on residual (dV/dt) */
            ResVx =  (-KVx[ix] + Fvx[ix])/rho_ML[ix];
            ResVy =  (-KVy[ix] + Fvy[ix])/rho_ML[ix];
            dVxdt[ix] = dVxdt[ix]*damp + ResVx;
            dVydt[ix] = dVydt[ix]*damp + ResVy;
        }
    }
} 

/*------- PT3_c Inputs and Outputs -------*/
/* 
Inputs: ftype* eta_nbv
        ftype* weights
        ftype* resolx
        ftype* resoly
        ftype rho
        ftype eta_b
        int nbv

Outputs: ftype* dtVx, ftype* dtVy
*/ 

void PT3_c(ftype* eta_nbv, ftype* weights, ftype* resolx, ftype* resoly, ftype rho, ftype eta_b, int nbv, ftype* dtVx, ftype* dtVy) {

    // The OpenACC copyin() clause is used to specify data that needs to be copied from the CPU to the GPU before kernel execution
    #pragma acc data copyin(eta_nbv[nbv], weights[nbv], dtVx[nbv], dtVy[nbv], resolx[nbv], resoly[nbv])
    {
        /* The OpenACC #pragma acc parallel loop directive instructs the compiler to parallelize the loop immediately following it */
        #pragma acc parallel loop async(2)
        for (int ix = 0; ix < nbv; ix += 1) {
            /* 2. Explicit CFL time step for viscous flow, x and y directions */
            eta_nbv[ix] = eta_nbv[ix]/weights[ix];
            ftype tmp_4eta = 4*eta_nbv[ix]*(1.+eta_b)*4.1;
            dtVx[ix] = rho*resolx[ix]*resolx[ix]/tmp_4eta;  
            dtVy[ix] = rho*resoly[ix]*resoly[ix]/tmp_4eta; 
        }
    }
} 

/*------- PT3_d Inputs and Outputs -------*/
/* 
Inputs: ftype* dVxdt
        ftype* dVydt
        ftype* vx
        ftype* vy
        ftype* spcvx
        ftype* spcvy
        ftype relaxation
        int nbv

Outputs: ftype* dVxdt
         ftype* dVydt
         ftype* vx
         ftype* vy
*/ 

void PT3_d(ftype* dVxdt, ftype* dVydt, ftype* vx, ftype* vy, ftype* spcvx, ftype* spcvy, ftype relaxation, int nbv, ftype* dtVx, ftype* dtVy) {

    /* The OpenACC copyin() clause is used to specify data that needs to be copied from the CPU to the GPU before kernel execution */
    #pragma acc data copyin(vx[nbv], vy[nbv], dVxdt[nbv], dVydt[nbv], dtVx[nbv], dtVy[nbv], spcvx[nbv], spcvy[nbv])
    {
        /* The OpenACC #pragma acc parallel loop directive instructs the compiler to parallelize the loop immediately following it */
        #pragma acc parallel loop async(1)
        for (int ix = 0; ix < nbv; ix += 1) {
            /* 3. velocity update, vx(new) = vx(old) + change in vx, Similarly for vy */
            vx[ix] = vx[ix] + relaxation*dVxdt[ix]*dtVx[ix];
            vy[ix] = vy[ix] + relaxation*dVydt[ix]*dtVy[ix];

            /* Apply Dirichlet boundary condition */
            if (!isnan(spcvx[ix])) {
                vx[ix]    = spcvx[ix];
                dVxdt[ix] = 0.;
            }
            if (!isnan(spcvy[ix])) {
                vy[ix]    = spcvy[ix];
                dVydt[ix] = 0.;
            }
        }
    }
}

/* MAIN */
int main() {

    /* Open input binary file */
    const char* inputfile  = "./JKS8e4.bin";
    const char* outputfile = "./output.outbin";
    FILE* fid = fopen(inputfile, "rb");
    if (fid == NULL) std::cerr << "Could not open file " << inputfile << " for binary reading or writing";

    /* Get All we need from binary file */
    int   nbe, nbv, M, N;
    ftype g, rho, rho_w, yts;
    int*   index           = NULL;
    ftype* spcvx           = NULL;
    ftype* spcvy           = NULL;
    ftype* x               = NULL;
    ftype* y               = NULL;
    ftype* H               = NULL;
    ftype* surface         = NULL;
    ftype* base            = NULL;
    ftype* ice_levelset    = NULL;
    ftype* ocean_levelset  = NULL;
    ftype* rheology_B_temp = NULL;
    ftype* vx              = NULL;
    ftype* vy              = NULL;
    ftype* friction        = NULL;
    FetchData(fid,&nbe,"md.mesh.numberofelements");
    FetchData(fid,&nbv,"md.mesh.numberofvertices");
    FetchData(fid,&g,"md.constants.g");
    FetchData(fid,&rho,"md.materials.rho_ice");
    FetchData(fid,&rho_w,"md.materials.rho_water");
    FetchData(fid,&yts,"md.constants.yts");
    FetchData(fid,&index,&M,&N,"md.mesh.elements");
    FetchData(fid,&spcvx,&M,&N,"md.stressbalance.spcvx");
    FetchData(fid,&spcvy,&M,&N,"md.stressbalance.spcvy");
    FetchData(fid,&x,&M,&N,"md.mesh.x");
    FetchData(fid,&y,&M,&N,"md.mesh.y");
    FetchData(fid,&H,&M,&N,"md.geometry.thickness");
    FetchData(fid,&surface,&M,&N,"md.geometry.surface");
    FetchData(fid,&base,&M,&N,"md.geometry.base");
    FetchData(fid,&ice_levelset,&M,&N,"md.mask.ice_levelset");
    FetchData(fid,&ocean_levelset,&M,&N,"md.mask.ocean_levelset");
    FetchData(fid,&rheology_B_temp,&M,&N,"md.materials.rheology_B");
    FetchData(fid,&vx,&M,&N,"md.initialization.vx");
    FetchData(fid,&vy,&M,&N,"md.initialization.vy");
    FetchData(fid,&friction,&M,&N,"md.friction.coefficient");

    /* Close input file */
    if (fclose(fid) != 0) std::cerr << "Could not close file " << inputfile;

    /* Constants */
    ftype n_glen     = 3.;
    ftype damp       = 0.98;  
    ftype rele       = 0.03;  
    ftype eta_b      = 0.5;
    ftype eta_0      = 1.e+14/2.;
    int   niter      = 5e6; 
    int   nout_iter  = 100; 
    ftype epsi       = 3.171e-7;
    ftype relaxation = 0.99; 
        
    /* Ceiling division to get the close to optimal GRID size */
    unsigned int GRID_Xe = 1 + ((nbe - 1) / BLOCK_Xe);
    unsigned int GRID_Xv = 1 + ((nbv - 1) / BLOCK_Xv);

    std::cout << "GRID_Xe = " << GRID_Xe << std::endl;
    std::cout << "GRID_Xv = " << GRID_Xv << std::endl;

    /* Set up GPU */
    int gpu_id = -1;
    dim3 gridv, blockv;
    dim3 gride, blocke;
    blockv.x = BLOCK_Xv; gridv.x = GRID_Xv;
    blocke.x = BLOCK_Xe; gride.x = GRID_Xe;
    
    /* OpenACC will automatically handle this, can optionally look into further for L1 (OpenACC has a function for this called cache()) **FIX** */
    /* 
    gpu_id = GPU_ID; cudaSetDevice(gpu_id); cudaGetDevice(&gpu_id);
    cudaDeviceReset(); cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);  // set L1 to preferred
    printf("Process uses GPU with id %d.\n", gpu_id);
    */

    /* Initial guesses (except vx and vy that we already loaded) */
    ftype* etan = new ftype[nbe];
    for (int i = 0; i < nbe; i++) etan[i] = 1.e+14;
    ftype* dVxdt = new ftype[nbv];
    for (int i = 0; i < nbv; i++) dVxdt[i] = 0.;
    ftype* dVydt = new ftype[nbv];
    for (int i = 0; i < nbv; i++) dVydt[i] = 0.;

    /* Add dtVx and dtVy */
    ftype* dtVx = new ftype[nbv];
    for (int i = 0; i < nbv; i++) dtVx[i] = 0.;
    ftype* dtVy = new ftype[nbv];
    for (int i = 0; i < nbv; i++) dtVy[i] = 0.;

    /* Manage derivatives once for all */
    ftype* alpha   = NULL;
    ftype* beta    = NULL;
    ftype* areas   = NULL;
    ftype* weights = NULL;
    NodalCoeffs(&areas, &alpha, &beta, index, x, y, nbe);
    Weights(&weights, index,a reas, nbe, nbv);

    /* MeshSize */
    ftype* resolx = new ftype[nbv];
    ftype* resoly = new ftype[nbv];
    MeshSize(resolx, resoly, index, x, y, areas, weights, nbe, nbv);

    /* Physical properties once for all */
    ftype* dsdx = new ftype[nbe];
    ftype* dsdy = new ftype[nbe];
    derive_xy_elem(dsdx, dsdy, surface, index, alpha, beta, nbe);
    ftype* Helem      = new ftype[nbe];
    ftype* rheology_B = new ftype[nbe];

    for (int i = 0; i < nbe; i++) {
        Helem[i]      = 1./3. * (H[index[i*3+0]-1] + H[index[i*3+1]-1] + H[index[i*3+2]-1]);
        rheology_B[i] = 1./3. * (rheology_B_temp[index[i*3+0]-1] + rheology_B_temp[index[i*3+1]-1] + rheology_B_temp[index[i*3+2]-1]);
    }

    /* Initial viscosity */
    ftype* dvxdx = new ftype[nbe];
    ftype* dvxdy = new ftype[nbe];
    ftype* dvydx = new ftype[nbe];
    ftype* dvydy = new ftype[nbe];

    derive_xy_elem(dvxdx, dvxdy, vx, index, alpha, beta, nbe);
    derive_xy_elem(dvydx, dvydy, vy, index, alpha, beta, nbe);

    /* Initializing variables for GPU residence only */
    ftype* KVx = new ftype[nbv];
    ftype* KVy = new ftype[nbv];
    ftype* eta_nbv = new ftype[nbv];
    ftype* Eta_nbe = new ftype[3*nbe];
    ftype* kvx = new ftype[3*nbe];
    ftype* kvy = new ftype[3*nbe];

    for (int i = 0; i < nbe; i++) {
        ftype eps_xx = dvxdx[i];
        ftype eps_yy = dvydy[i];
        ftype eps_xy = .5 * (dvxdy[i] + dvydx[i]);
        ftype EII2 = pow(eps_xx, 2) + pow(eps_yy, 2) + pow(eps_xy, 2) + eps_xx * eps_yy;
        ftype eta_it = 1.e + 14/2.;

        if (EII2 > 0.) eta_it = rheology_B[i]/(2*pow(EII2, (n_glen-1.)/(2*n_glen)));

        etan[i] = min(eta_it, eta_0 * 1e5);
        if (isnan(etan[i])) { std::cerr << "Found NaN in etan[i]"; return 1; }
    }

    /* Linear integration points order 3 */
    ftype wgt3[] = { 0.555555555555556, 0.888888888888889, 0.555555555555556};
    ftype xg3[]  = {-0.774596669241483, 0.000000000000000, 0.774596669241483};

    /* Compute RHS amd ML once for all */
    ftype* ML            = new ftype[nbv];
    ftype* Fvx           = new ftype[nbv];
    ftype* Fvy           = new ftype[nbv];
    ftype* groundedratio = new ftype[nbe];
    bool*  isice         = new bool[nbe];     
    ftype level[3];      

    for (int i = 0; i < nbv; i++) {
        ML[i]  = 0.;
        Fvx[i] = 0.;
        Fvy[i] = 0.;
    }
    for (int n = 0; n < nbe; n++) {
        /* Lumped mass matrix */
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (i == j)
                ML[index[n*3+j]-1] += areas[n]/6.;
                else
                ML[index[n*3+j]-1] += areas[n]/12.;
            }
        }
        /* Is there ice at all in the current element? */
        level[0] = ice_levelset[index[n*3+0]-1];
        level[1] = ice_levelset[index[n*3+1]-1];
        level[2] = ice_levelset[index[n*3+2]-1];
        if (level[0] < 0 || level[1] < 0 || level[2] < 0) { isice[n] = true; }
        else {
            isice[n] = false;

            for(int i = 0; i < 3; i++) {
                vx[index[n*3+i]-1] = 0.;
                vy[index[n*3+i]-1] = 0.;
            }
            continue;
        }
        /* RHS, 'F ' Driving Stress */
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (i == j) {
                    Fvx[index[n*3+i]-1] += -rho*g*H[index[n*3+j]-1]*dsdx[n]*areas[n]/6.;
                    Fvy[index[n*3+i]-1] += -rho*g*H[index[n*3+j]-1]*dsdy[n]*areas[n]/6.;
                }
                else {
                    Fvx[index[n*3+i]-1] += -rho*g*H[index[n*3+j]-1]*dsdx[n]*areas[n]/12.;
                    Fvy[index[n*3+i]-1] += -rho*g*H[index[n*3+j]-1]*dsdy[n]*areas[n]/12.;
                }
            }
        }
    }
    /* RHS (Water pressure at the ice front) */
    for (int n = 0; n < nbe; n++) {
        /*Determine if there is an ice front*/
        level[0] = ice_levelset[index[n*3+0]-1];
        level[1] = ice_levelset[index[n*3+1]-1];
        level[2] = ice_levelset[index[n*3+2]-1];
        int count = 0;

        for (int i = 0; i < 3; i++) if (level[i] < 0.) count++;

        if (count == 1) {
            /* Ok this element has an ice front, get indices of the 2 vertices */
            int seg1[2] = {index[n*3+0]-1,index[n*3+1]-1};
            int seg2[2] = {index[n*3+1]-1,index[n*3+2]-1};
            int seg3[2] = {index[n*3+2]-1,index[n*3+0]-1};
            int pairids[2];
            if (ice_levelset[seg1[0]] >= 0 && ice_levelset[seg1[1]] >= 0) {
                pairids[0] = seg1[0]; pairids[1] = seg1[1];
            }
            else if (ice_levelset[seg2[0]] >= 0 && ice_levelset[seg2[1]] >= 0) {
                pairids[0] = seg2[0]; pairids[1] = seg2[1];
            }
            else if (ice_levelset[seg3[0]] >= 0 && ice_levelset[seg3[1]] >= 0) {
                pairids[0] = seg3[0]; pairids[1] = seg3[1];
            }
            else { std::cerr << "Case not supported"; }
            /* Get normal */
            ftype len = sqrt(pow(x[pairids[1]]-x[pairids[0]], 2) + pow(y[pairids[1]]-y[pairids[0]], 2) );
            ftype nx  = +(y[pairids[1]]-y[pairids[0]])/len;
            ftype ny  = -(x[pairids[1]]-x[pairids[0]])/len;

            /* RHS */
            for (int gg = 0; gg < 2; gg++) {
                ftype phi1 = (1.0 -xg3[gg])/2.;
                ftype phi2 = (1.0 +xg3[gg])/2.;
                ftype bg = base[pairids[0]] * phi1 + base[pairids[1]] * phi2;
                ftype Hg = H[pairids[0]] * phi1 + H[pairids[1]] * phi2;
                bg = min(bg, 0.0);
                Fvx[pairids[0]] = Fvx[pairids[0]] + wgt3[gg]/2*1/2*(-rho_w*g*pow(bg,2)+rho*g*pow(Hg,2))*nx*len*phi1;
                Fvx[pairids[1]] = Fvx[pairids[1]] + wgt3[gg]/2*1/2*(-rho_w*g*pow(bg,2)+rho*g*pow(Hg,2))*nx*len*phi2;
                Fvy[pairids[0]] = Fvy[pairids[0]] + wgt3[gg]/2*1/2*(-rho_w*g*pow(bg,2)+rho*g*pow(Hg,2))*ny*len*phi1;
                Fvy[pairids[1]] = Fvy[pairids[1]] + wgt3[gg]/2*1/2*(-rho_w*g*pow(bg,2)+rho*g*pow(Hg,2))*ny*len*phi2;
            } 
        }
        /* One more thing in this element loop: prepare groundedarea needed later for the calculation of basal friction */
        level[0] = ocean_levelset[index[n*3+0]-1];
        level[1] = ocean_levelset[index[n*3+1]-1];
        level[2] = ocean_levelset[index[n*3+2]-1];

        if (level[0] >= 0. && level[1] >= 0. && level[2] >= 0.) {
            /* Completely grounded */
            groundedratio[n] = 1.;
        }
        else if (level[0] <= 0. && level[1] <= 0. && level[2] <= 0.) {
            /* Completely floating */
            groundedratio[n] = 0.;
        }
        else {
            /* Partially floating, */
            ftype s1, s2;
            /* Nodes 0 and 1 are similar, so points must be found on segment 0-2 and 1-2 */
            if (level[0] * level[1] > 0) {
                s1 = level[2]/(level[2]-level[1]);
                s2 = level[2]/(level[2]-level[0]);
            }
            /* Nodes 1 and 2 are similar, so points must be found on segment 0-1 and 0-2 */
            else if (level[1] * level[2] > 0) {
                s1 = level[0]/(level[0]-level[1]);
                s2 = level[0]/(level[0]-level[2]);
            }
            /* Nodes 0 and 2 are similar, so points must be found on segment 1-0 and 1-2 */
            else if (level[0] * level[2] > 0) {
                s1 = level[1]/(level[1]-level[0]);
                s2 = level[1]/(level[1]-level[2]);
            }
            else { std::cerr << "Should not be here..."; }

            if (level[0] * level[1] * level[2] > 0.) {
                /* Two nodes floating, inner triangle is grounded */
                groundedratio[n]= s1 * s2;
            }
            else {
                /* One node floating, inner triangle is floating */
                groundedratio[n]= (1.-s1 * s2);
            }
        }
    }

    /* Finally add calculation of friction coefficient */
    ftype* alpha2 = new ftype[nbv];

    for (int i = 0; i < nbv; i++) {
        /* Compute effective pressure */
        ftype p_ice   = g * rho * H[i];
        ftype p_water = -rho_w * g * base[i];
        ftype Neff    = p_ice - p_water;
        if (Neff < 0.) Neff = 0.;
        /* Compute alpha2 */
        alpha2[i] = pow(friction[i], 2) * Neff;
    }

    /* Prepare head and next vectors for chain algorithm, at this point we have not seen any of the elements, so just set the head to -1 (=stop) */
    int* head = new int[nbv];
    int* next  = new int[3 * nbe];

    for (int i = 0; i < nbv; i++) head[i] = -1;

    /* Now construct the chain */
    for (int k = 0; k < nbe; k++) {
        for (int j = 0; j < 3; j++) {
            int i;
            /* Unique linear index of current vertex in index */
            int p = 3 * k + j;
            i = index[p];
            next[p] = head[i - 1];
            head[i - 1] = p + 1;
        }
    }

    /* Note: Index array starts at 0, but the node# starts at 1 */
    /* Now we can construct the connectivity matrix */
    int MAXCONNECT = 8;
    int* connectivity = new int[nbv*MAXCONNECT];
    int* columns = new int[nbv*MAXCONNECT];

    for (int i = 0; i < nbv; i++) {
        /* Go over all of the elements connected to node I */
        int count = 0;
        int p = head[i];
        // for (int p = head[i]; p != -1; p = next[p]) {
        while (p != -1) {
            /* ”row" in index */
            int k = p / 3 + 1;
            /* "column" in index */
            int j = (p % 3) - 1;

            if (j == -1) {
                j = 2;
                k = k - 1;
            }
            /* Sanity check */
            if (index[p - 1] != i + 1) {
                std::cout << "Error occurred" << std::endl;;
            }
            /* Enter element in connectivity matrix */
            connectivity[i * MAXCONNECT + count] = k;
            columns[i * MAXCONNECT + count] = j;
            count++;
            p = next[p - 1];
        }
    }
  
    ftype* device_maxvalx = new ftype[GRID_Xv];
    ftype* device_maxvaly = new ftype[GRID_Xv];
    for (int i = 0; i < GRID_Xv; i++) device_maxvalx[i] = 0.;
    for (int i = 0; i < GRID_Xv; i++) device_maxvaly[i] = 0.;

    ftype* rho_ML = new ftype[nbv];

    for (int i = 0; i < nbv ; i++) {
	    rho_ML[i] = rho * max(60.0, H[i]) * ML[i];
    }

    ftype* gr_a_alpha2 = new ftype[nbe * 3];

    for (int i = 0;i < nbe; i++) {
        if (groundedratio[i] > 0.) {
            int n3 = i * 3;
            ftype tmp_gr_a = groundedratio[i] * areas[i];

            for (int j = 0; j < 3; j++) {
                 int i_index = index[n3 + j] - 1;
                 gr_a_alpha2[n3 + j] =  tmp_gr_a * alpha2[i_index];
            }
        }
    }

    /* Performance metrics (timing) */
    ftype time_s = 0.0;
    ftype mem = (ftype)1e-9 * (ftype)nbv * sizeof(ftype);
    int nIO = 8;

    /* Main Loop */
    std::cout << "Starting PT loop, nbe = " << nbe << ", nbv = " << nbv << std::endl; 
    int iter;
    ftype iterror;

    // OpenACC enter data construct copyin() copies in all relevant vectors from host (CPU) to device (GPU) and create() allocates relevant vectors on device (GPU)
    #pragma acc data enter copyin(index, rho_ML, gr_a_alpha2, vx, vy, alpha, beta, etan, rheology_B, Helem, areas, weights, ML, Fvx, Fvy, dVxdt, dVydt, dtVx, dtVy, resolx, resoly, H, spcvx, spcvy, alpha2, groundedratio, isice, connectivity, columns, device_maxvalx, device_maxvaly) create(dvxdx, dvxdy, dvydx, dvydy, KVx, KVy, eta_nbv, Eta_nbe, kvx, kvy)

    for (iter = 1; iter <= niter; iter++) {
      
        /* Calling kernels here as normal function calls with NVTX annotations */
        if (iter == 11)

            nvtxEventAttributes_t eventAttrib_PT1 = {0};
            eventAttrib_PT1.version = NVTX_VERSION;
            eventAttrib_PT1.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            eventAttrib_PT1.colorType = NVTX_COLOR_ARGB;
            eventAttrib_PT1.color = 0xFFFFC0CB;
            eventAttrib_PT1.messageType = NVTX_MESSAGE_TYPE_ASCII;
            eventAttrib_PT1.message.ascii = "PT1";

            nvtxRangePushEx(&eventAttrib_PT1);

            PT1(vx, vy, alpha, beta, index, kvx, kvy, etan, Helem, areas, isice, Eta_nbe, rheology_B, n_glen, eta_0, rele, nbe);
            
            nvtxRangePop();

            nvtxEventAttributes_t eventAttrib_PT2_xy = {0};
            eventAttrib_PT2_xy.version = NVTX_VERSION;
            eventAttrib_PT2_xy.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            eventAttrib_PT2_xy.colorType = NVTX_COLOR_ARGB;
            eventAttrib_PT2_xy.color = 0xFF0000FF;
            eventAttrib_PT2_xy.messageType = NVTX_MESSAGE_TYPE_ASCII;
            eventAttrib_PT2_xy.message.ascii = "PT2_xy";
            
            nvtxRangePushEx(&eventAttrib_PT2_xy);

            PT2_x(kvx, groundedratio, areas, index, alpha2, vx, gr_a_alpha2, isice, nbe);   
            PT2_y(kvy, groundedratio, areas, index, alpha2, vy, gr_a_alpha2, isice, nbe);
            
            nvtxRangePop();

            nvtxEventAttributes_t eventAttrib_PT3_a = {0};
            eventAttrib_PT3_a.version = NVTX_VERSION;
            eventAttrib_PT3_a.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            eventAttrib_PT3_a.colorType = NVTX_COLOR_ARGB;
            eventAttrib_PT3_a.color = 0xFF800080;
            eventAttrib_PT3_a.messageType = NVTX_MESSAGE_TYPE_ASCII;
            eventAttrib_PT3_a.message.ascii = "PT3_a";
            
            nvtxRangePushEx(&eventAttrib_PT3_a);

            PT3_a(kvx, kvy, Eta_nbe, eta_nbv, connectivity, columns, KVx, KVy, nbv);
            
            nvtxRangePop();

            /* Define NVTX event attributes: PT3_bc, cyan */
            nvtxEventAttributes_t eventAttrib_PT3_bc = {0};
            eventAttrib_PT3_bc.version = NVTX_VERSION;
            eventAttrib_PT3_bc.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            eventAttrib_PT3_bc.colorType = NVTX_COLOR_ARGB;
            eventAttrib_PT3_bc.color = 0xFF00FFFF;
            eventAttrib_PT3_bc.messageType = NVTX_MESSAGE_TYPE_ASCII;
            eventAttrib_PT3_bc.message.ascii = "PT3_bc";
            
            nvtxRangePushEx(&eventAttrib_PT3_bc);

            PT3_b(KVx, KVy, Fvx, Fvy, dVxdt, dVydt, rho_ML, damp, nbv);
            PT3_c(eta_nbv, weights, resolx, resoly, rho, eta_b, nbv, dtVx, dtVy);
            
            nvtxRangePop();
            
            nvtxEventAttributes_t eventAttrib_PT3_d = {0};
            eventAttrib_PT3_d.version = NVTX_VERSION;
            eventAttrib_PT3_d.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            eventAttrib_PT3_d.colorType = NVTX_COLOR_ARGB;
            eventAttrib_PT3_d.color = 0xFFFFA500;
            eventAttrib_PT3_d.messageType = NVTX_MESSAGE_TYPE_ASCII;
            eventAttrib_PT3_d.message.ascii = "PT3_d";

            nvtxRangePushEx(&eventAttrib_PT3_d);

            PT3_d(dVxdt, dVydt, vx, vy, spcvx, spcvy, relaxation, nbv, dtVx, dtVy); 
            
            nvtxRangePop();

        /* This if statement is just printing iteration and error values, specifically GPU functions **FIX** */
        /* if ((iter % nout_iter) == 0) {

            // Get final error estimate 
            __device_max_x(dVxdt); 
            __device_max_y(dVydt);

            if (isnan(device_MAXx) || isnan(device_MAXy)) { break; }
            else {
                iterror = max(device_MAXx, device_MAXy);
                std::cout << "iter=" << iter << ", err=" << iterror << std::endl;
                if ((iterror < epsi) && (iter > 100)) break;
            } 
        }
        */  
    }
    /* OpenACC exit data construct copies out all relevant vectors */
    #pragma acc data exit copyout(vx, vy)

    /* We will need to implement a CPU version for timing analysis **FIX** */
    /* 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float seconds = 0;
    cudaEventElapsedTime(&seconds, start, stop);
    seconds = seconds/1000.0;
    time_s = seconds; ftype gbs = mem/time_s;
    std::cout << "\n Perf: " << time_s << " sec. (@ " << gbs*(iter-10)*nIO << " GB/s)" << std::endl;
    */

    /* Write output */
    fid = fopen(outputfile,"wb");
    if (fid==NULL) std::cerr << "Could not open file " << outputfile << " for binary reading or writing";
    WriteData(fid, "PTsolution", "SolutionType");
    WriteData(fid, vx, nbv, 1, "Vx");
    WriteData(fid, vy, nbv, 1, "Vy");
    if (fclose(fid)!=0) std::cerr << "Could not close file " << outputfile;

    /* Cleanup and return */
    delete [] index;
    delete [] x;
    delete [] y;
    delete [] H;
    delete [] surface;
    delete [] base;
    delete [] spcvx;
    delete [] spcvy;
    delete [] ice_levelset;
    delete [] ocean_levelset;
    delete [] rheology_B;
    delete [] rheology_B_temp;
    delete [] vx;
    delete [] vy;
    delete [] friction;
    delete [] alpha2;
    delete [] etan;
    delete [] dVxdt;
    delete [] dVydt;
    delete [] dtVx;
    delete [] dtVy;
    delete [] alpha;
    delete [] beta;
    delete [] areas;
    delete [] weights;
    delete [] resolx;
    delete [] resoly;
    delete [] dsdx;
    delete [] dsdy;
    delete [] Helem;
    delete [] ML;
    delete [] Fvx;
    delete [] Fvy;
    delete [] rho_ML;
    delete [] gr_a_alpha2;
    delete [] KVx;
    delete [] KVy;
    delete [] eta_nbv;
    delete [] Eta_nbe;
    delete [] kvx;
    delete [] kvy;

    return 0;
}