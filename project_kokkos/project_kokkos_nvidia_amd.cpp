
#include <iostream>
#include <Kokkos_Core.hpp>
#include <string.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include <chrono>
//when runing on the nvidia change memory space to Kokkos::CudaSpace and in AMD change to Kokkos::HIPSpace//
#define vector2(name,type,size) Kokkos::View<type *, Kokkos::HIPSpace> name(#name,size)
#define STRING(x) #x
#define vector(orig,type,size) Kokkos::View<type *, Kokkos::HIPSpace > d_ ##orig (STRING(d_ ##orig),size);\
 Kokkos::View<type *>::HostMirror h_ ##orig = Kokkos::create_mirror_view(d_ ##orig);\
  for(int i=0;i< size ;i++){h_ ##orig[i]= orig[i];};\
   Kokkos::deep_copy(d_ ##orig, h_ ##orig);

//Define GPU specific variable
#define GPU_ID   0

#define BLOCK_Xe  128
#define BLOCK_Xv  128

#define blockid (blockidx.x)
#define threadid (threadidx.x)
#define isBlockMaster (threadldx.x==0)

#include "helpers.h"


#define div10  0.1
#define div30  0.0333333333
#define div60  0.0166666667


void PT1(Kokkos::View<ftype*> vx, Kokkos::View<ftype*> vy, Kokkos::View<ftype*> alpha, Kokkos::View<ftype*> beta, Kokkos::View<int*> index, Kokkos::View<ftype*> kvx, Kokkos::View<ftype*> kvy, Kokkos::View<ftype*> etan, Kokkos::View<ftype*> Helem, Kokkos::View<ftype*> areas, Kokkos::View<bool*> isice, Kokkos::View<ftype*> Eta_nbe, Kokkos::View<ftype*> rheology_B, ftype n_glen, ftype eta_0, ftype rele, int nbe)
{

auto do_filtering = KOKKOS_LAMBDA(int64_t const ix) {
        
        if (isice[ix]){

            ftype Localalpha[3];
            ftype Localbeta[3];
            ftype Localvx[3];
            ftype Localvy[3];
        
            for(int i=0; i<3; i++){
            Localalpha[i] =  alpha[ix*3+i];
            Localbeta[i] =   beta[ix*3+i];
            Localvx[i] =  vx[index[ix*3+i]-1];
            Localvy[i] =  vy[index[ix*3+i]-1];
            }
        
            ftype dvxdx =  Localvx[0]*Localalpha[0] + Localvx[1]*Localalpha[1] + Localvx[2]*Localalpha[2];
            ftype dvxdy =  Localvx[0]* Localbeta[0] + Localvx[1]* Localbeta[1] + Localvx[2]* Localbeta[2];
            ftype dvydx =  Localvy[0]*Localalpha[0] + Localvy[1]*Localalpha[1] + Localvy[2]*Localalpha[2];
            ftype dvydy =  Localvy[0]* Localbeta[0] + Localvy[1]* Localbeta[1] + Localvy[2]* Localbeta[2];

            ftype  eps_xx = dvxdx;
            ftype  eps_yy = dvydy;
            ftype  eps_xy = .5*(dvxdy+dvydx);
            ftype  EII2   = eps_xx*eps_xx + eps_yy*eps_yy + eps_xy*eps_xy + eps_xx*eps_yy;
            ftype  eta_it = 5e+13;

         if (EII2>0.) eta_it = rheology_B[ix]/(2*Kokkos::powf((ftype)EII2,(ftype)(n_glen-1.)/(2*n_glen)));

        
                etan[ix] = Kokkos::min(ftype(Kokkos::expf(rele*Kokkos::logf(eta_it) + (1.0-rele)*Kokkos::logf(etan[ix]))),ftype(eta_0*1e5));

                //Viscous Deformation//
                ftype tmp_2hele_etan_areas = 2 * Helem[ix] * etan[ix] *areas[ix];
                for (int i = 0; i < 3; i++){
                        kvx[ix*3+i] = tmp_2hele_etan_areas * ((2 * eps_xx + eps_yy) * Localalpha[i]   +  eps_xy *  Localbeta[i] );
                        kvy[ix*3+i] = tmp_2hele_etan_areas * ((2 * eps_yy + eps_xx) * Localbeta[i]    +  eps_xy * Localalpha[i] );            
            }
        }
   
        Eta_nbe[ix] = etan[ix]*areas[ix];
    };

    Kokkos::parallel_for("PT1", nbe, do_filtering);
}

// Kokkos Views for the input arrays
void PT2_x(Kokkos::View<ftype*> kvx, Kokkos::View<ftype*> groundedratio, Kokkos::View<ftype*> areas, Kokkos::View<int*> index, Kokkos::View<ftype*> alpha2, Kokkos::View<ftype*> vx, Kokkos::View<ftype*> gr_a_alpha2, Kokkos::View<bool*> isice, int nbe){

auto do_filtering = KOKKOS_LAMBDA(int64_t const ix) {

 /*Add basal friction*/
        if (groundedratio[ix] > 0.){
            int n3 = ix * 3;

       ftype myLocalIndex[3][3];
            for (int i = 0; i < 3; i++){
              	for (int j = 0; j < 3; j++){
                      int j_index = index[n3 + j] - 1;
                      myLocalIndex[i][j] = gr_a_alpha2[n3 + i] * vx[j_index];
                }
            }


            ftype tempOutput[3];
            for (int k = 0; k < 3; k++){
                tempOutput[k] = kvx[n3 + k];
            }   


            ftype division;
            for (int k = 0; k < 3; k++){
                for (int i = 0; i < 3; i++){
                    for (int j = 0; j < 3; j++){
                      	ftype temp = myLocalIndex[i][j];

                        if (i == j && j == k){
			    division = div10;
                        } else if ((i!=j) && (j!=k) && (k!=i)){
			    division = div60;
                        } else{
			    division = div30;
                        }
			tempOutput[k] = isice[ix] * tempOutput[k] + temp*division;
                    }
                }
            } 
        
             for (int k = 0; k < 3; k++){
                kvx[n3 + k] = tempOutput[k];
            }  
        }
    };
       
        Kokkos::parallel_for("PT2_x", nbe, do_filtering);
}

void PT2_y(Kokkos::View<ftype*> kvy, Kokkos::View<ftype*> groundedratio, Kokkos::View<ftype*> areas, Kokkos::View<int*> index, Kokkos::View<ftype*> alpha2, Kokkos::View<ftype*> vy, Kokkos::View<ftype*> gr_a_alpha2, Kokkos::View<bool*> isice, int nbe){

auto do_filtering = KOKKOS_LAMBDA(int64_t const ix) {

/*Add basal friction*/
        if (groundedratio[ix] > 0.){
            int n3 = ix * 3;

            ftype myLocalIndex[3][3];
            for (int i = 0; i < 3; i++){
                for (int j = 0; j < 3; j++){
                      int j_index = index[n3 + j] - 1;
                      myLocalIndex[i][j] = gr_a_alpha2[n3 + i] * vy[j_index];
                }
            }

          ftype tempOutput[3];
            for (int k = 0; k < 3; k++){
                tempOutput[k] = kvy[n3 + k];
            }

            ftype division;
            for (int k = 0; k < 3; k++){
                for (int i = 0; i < 3; i++){
                    for (int j = 0; j < 3; j++){
                      	ftype temp = myLocalIndex[i][j];

                        if (i == j && j == k){
			    division = div10;
                        } else if ((i!=j) && (j!=k) && (k!=i)){
			    division = div60;
                        } else{
			    division = div30;
                        }
			tempOutput[k] = isice[ix] * tempOutput[k] + temp*division;
                    }
                }
            } 

              for (int k = 0; k < 3; k++){
                kvy[n3 + k] = tempOutput[k];
            }  
  

        }
    };
            Kokkos::parallel_for("PT2_y", nbe, do_filtering);

}

void PT3(Kokkos::View<ftype*> kvx, Kokkos::View<ftype*> kvy, Kokkos::View<ftype*> Eta_nbe, Kokkos::View<ftype*> eta_nbv, Kokkos::View<int*> connectivity, Kokkos::View<int*> columns, Kokkos::View<ftype*> weights, Kokkos::View<ftype*> ML, Kokkos::View<ftype*> KVx, Kokkos::View<ftype*> KVy, Kokkos::View<ftype*> Fvx, Kokkos::View<ftype*> Fvy, Kokkos::View<ftype*> dVxdt, Kokkos::View<ftype*> dVydt, Kokkos::View<ftype*> resolx, Kokkos::View<ftype*> resoly, Kokkos::View<ftype*> H, Kokkos::View<ftype*> vx, Kokkos::View<ftype*> vy, Kokkos::View<ftype*> spcvx, Kokkos::View<ftype*> spcvy, Kokkos::View<ftype*> rho_ML, ftype rho, ftype damp, ftype relaxation, ftype eta_b, int nbv){ 

auto do_filtering = KOKKOS_LAMBDA(int64_t const ix) {

    ftype ResVx;
    ftype ResVy;
    ftype dtVx;
    ftype dtVy;

        KVx[ix] = 0.;
        KVy[ix] = 0.;

        int localColumns[8];
        int localConnectivity[8];
        for(int j=0; j<8; j++){
	        localConnectivity[j] = connectivity[(ix * 8  + j)];
	        localColumns[j] = columns[(ix * 8  + j)];
        } 
      
	    
           ftype tmp_KVx = KVx[ix];
           ftype tmp_KVy = KVy[ix];
           ftype tmp_eta_nbv = eta_nbv[ix];

          for(int j=0;j<8;j++){
        
            if (localConnectivity[j] != 0){
                tmp_KVx = tmp_KVx + kvx[((localConnectivity[j])-1) *3 + ((localColumns[(j)]))];
                tmp_KVy = tmp_KVy + kvy[((localConnectivity[j])-1) *3 + ((localColumns[(j)]))];
            }
              
            if (localConnectivity[j] != 0){
                tmp_eta_nbv = tmp_eta_nbv + Eta_nbe[localConnectivity[j]-1];
            }
            else{break;}
        }    

        KVx[ix] = tmp_KVx;
        KVy[ix] = tmp_KVy;

        eta_nbv[ix] =tmp_eta_nbv/weights[ix];   

        /*1. Get time derivative based on residual (dV/dt)*/
	    ResVx =  (-KVx[ix] + Fvx[ix])/rho_ML[ix];
	    ResVy =  (-KVy[ix] + Fvy[ix])/rho_ML[ix];
        dVxdt[ix] = dVxdt[ix]*damp + ResVx;
        dVydt[ix] = dVydt[ix]*damp + ResVy;

    
        /*2. Explicit CFL time step for viscous flow, x and y directions*/
        ftype tmp_4eta = 4*eta_nbv[ix]*(1.+eta_b)*4.1;

        dtVx = rho*resolx[ix]*resolx[ix]/tmp_4eta;  
        dtVy = rho*resoly[ix]*resoly[ix]/tmp_4eta; 

        /*3. velocity update, vx(new) = vx(old) + change in vx, Similarly for vy*/
        vx[ix] = vx[ix] + relaxation*dVxdt[ix]*dtVx;
        vy[ix] = vy[ix] + relaxation*dVydt[ix]*dtVy;

        /*Apply Dirichlet boundary condition*/
        if (!Kokkos::isnan(spcvx[ix])){
            vx[ix]    = spcvx[ix];
             dVxdt[ix] = 0.;
        }
        if (!Kokkos::isnan(spcvy[ix])){
            vy[ix]    = spcvy[ix];
            dVydt[ix] = 0.;
        }
    };
                Kokkos::parallel_for("PT3", nbv, do_filtering);
}


int main(int argc, char* argv[]){

Kokkos::initialize(argc, argv);{

    /*Open input binary file*/
    const char* inputfile  = "./JKS8e4.bin";
    std::string str = argc >1? argv[1] : inputfile;
    const char* outputfile = "./output.outbin";
    FILE* fid = fopen(str.c_str(),"rb");
    if(fid==NULL) std::cerr<<"could not open file " << inputfile << " for binary reading or writing";

    /*Get All we need from binary file*/
    int    nbe,nbv,M,N;
    ftype g,rho,rho_w,yts;
    int    *index           = NULL;
    ftype *spcvx           = NULL;
    ftype *spcvy           = NULL;
    ftype *x               = NULL;
    ftype *y               = NULL;
    ftype *H               = NULL;
    ftype *surface         = NULL;
    ftype *base            = NULL;
    ftype *ice_levelset    = NULL;
    ftype *ocean_levelset  = NULL;
    ftype *rheology_B_temp = NULL;
    ftype *vx              = NULL;
    ftype *vy              = NULL;
    ftype *friction        = NULL;
    
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
 
    // Close input file
    if (fclose(fid) != 0) {
        std::cerr << "Could not close file " << inputfile << "\n";
        return 1;
    }

    // Constants
    ftype n_glen = 3.;
    ftype damp = std::stod(argv[2]);
    ftype rele = std::stod(argv[3]);
    ftype eta_b = 0.5;
    ftype eta_0 = 1.e+14 / 2.;
    int niter = 5e6;
    int nout_iter = 100;
    ftype epsi = 3.171e-7;
    ftype relaxation = std::stod(argv[4]);

    // Ceiling division to get the close to optimal GRID size
    unsigned int GRID_Xe = 1 + ((nbe - 1) / BLOCK_Xe);
    unsigned int GRID_Xv = 1 + ((nbv - 1) / BLOCK_Xv);

    std::cout << "GRID_Xe=" << GRID_Xe << std::endl;
    std::cout << "GRID_Xv=" << GRID_Xv << std::endl;

    // Set up GPU
    int gpu_id = -1;
    gpu_id = GPU_ID;
    printf("Process uses GPU with id %d.\n", gpu_id);

    /*Initial guesses (except vx and vy that we already loaded)*/
    ftype* etan = new ftype[nbe];
    for(int i=0;i<nbe;i++) etan[i] = 1.e+14;
    ftype* dVxdt = new ftype[nbv];
    for(int i=0;i<nbv;i++) dVxdt[i] = 0.;
    ftype* dVydt = new ftype[nbv];
    for(int i=0;i<nbv;i++) dVydt[i] = 0.;

    /*Manage derivatives once for all*/
    ftype* alpha   = NULL;
    ftype* beta    = NULL;
    ftype* areas   = NULL;
    ftype* weights = NULL;
    NodalCoeffs(&areas,&alpha,&beta,index,x,y,nbe);
    Weights(&weights,index,areas,nbe,nbv);

    /*MeshSize*/
    ftype* resolx = new ftype[nbv];
    ftype* resoly = new ftype[nbv];
    MeshSize(resolx,resoly,index,x,y,areas,weights,nbe,nbv);

    /*Physical properties once for all*/
    ftype* dsdx = new ftype[nbe];
    ftype* dsdy = new ftype[nbe];
    derive_xy_elem(dsdx,dsdy,surface,index,alpha,beta,nbe);
    ftype* Helem      = new ftype[nbe];
    ftype* rheology_B = new ftype[nbe];
    for(int i=0;i<nbe;i++){
        Helem[i]      = 1./3. * (H[index[i*3+0]-1] + H[index[i*3+1]-1] + H[index[i*3+2]-1]);
        rheology_B[i] = 1./3. * (rheology_B_temp[index[i*3+0]-1] + rheology_B_temp[index[i*3+1]-1] + rheology_B_temp[index[i*3+2]-1]);
    }

    //Initial viscosity//
    ftype* dvxdx   = new ftype[nbe];
    ftype* dvxdy   = new ftype[nbe];
    ftype* dvydx   = new ftype[nbe];
    ftype* dvydy   = new ftype[nbe];

    derive_xy_elem(dvxdx,dvxdy,vx,index,alpha,beta,nbe);
    derive_xy_elem(dvydx,dvydy,vy,index,alpha,beta,nbe);

    for(int i=0;i<nbe;i++){
        ftype eps_xx = dvxdx[i];
        ftype eps_yy = dvydy[i];
        ftype eps_xy = .5*(dvxdy[i]+dvydx[i]);
        ftype EII2 = Kokkos::pow(eps_xx,2) + Kokkos::pow(eps_yy,2) + Kokkos::pow(eps_xy,2) + eps_xx*eps_yy;
        ftype eta_it = 1.e+14/2.;
        if (EII2>0.) eta_it = rheology_B[i]/(2*Kokkos::pow(EII2,(n_glen-1.)/(2*n_glen)));

        etan[i] = Kokkos::min(eta_it,eta_0*1e5);
        if (Kokkos::isnan(etan[i])){ std::cerr<<"Found NaN in etan[i]"; return 1;}
    }

    /*Linear integration points order 3*/
    ftype wgt3[] = { 0.555555555555556, 0.888888888888889, 0.555555555555556 };
    ftype xg3[]  = {-0.774596669241483, 0.000000000000000, 0.774596669241483 };

    /*Compute RHS amd ML once for all*/
    ftype* ML            = new ftype[nbv];
    ftype* Fvx           = new ftype[nbv];
    ftype* Fvy           = new ftype[nbv];
    ftype* groundedratio = new ftype[nbe];
    bool*   isice         = new bool[nbe];     
    ftype level[3];      

    for(int i=0;i<nbv;i++){
        ML[i]  = 0.;
        Fvx[i] = 0.;
        Fvy[i] = 0.;
    }
    for(int n=0;n<nbe;n++){
        /*Lumped mass matrix*/
        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                if (i==j)
                 ML[index[n*3+j]-1] += areas[n]/6.;
                else
                 ML[index[n*3+j]-1] += areas[n]/12.;
            }
        }
        /*Is there ice at all in the current element?*/
        level[0] = ice_levelset[index[n*3+0]-1];
        level[1] = ice_levelset[index[n*3+1]-1];
        level[2] = ice_levelset[index[n*3+2]-1];
        if (level[0]<0 || level[1]<0 || level[2]<0){
            isice[n] = true;
        }
        else{
            isice[n] = false;
            for(int i=0;i<3;i++){
                vx[index[n*3+i]-1] = 0.;
                vy[index[n*3+i]-1] = 0.;
            }
            continue;
        }
        /*RHS, 'F ' Driving Stress*/
        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                if (i==j){
                    Fvx[index[n*3+i]-1] += -rho*g*H[index[n*3+j]-1]*dsdx[n]*areas[n]/6.;
                    Fvy[index[n*3+i]-1] += -rho*g*H[index[n*3+j]-1]*dsdy[n]*areas[n]/6.;
                }
                else{
                    Fvx[index[n*3+i]-1] += -rho*g*H[index[n*3+j]-1]*dsdx[n]*areas[n]/12.;
                    Fvy[index[n*3+i]-1] += -rho*g*H[index[n*3+j]-1]*dsdy[n]*areas[n]/12.;
                }
            }
        }
    }


    /*RHS (Water pressure at the ice front)*/
    for(int n=0;n<nbe;n++){
        /*Determine if there is an ice front*/
        level[0] = ice_levelset[index[n*3+0]-1];
        level[1] = ice_levelset[index[n*3+1]-1];
        level[2] = ice_levelset[index[n*3+2]-1];
        int count = 0;
        for(int i=0;i<3;i++) if (level[i]<0.) count++;
        if (count==1){
            /*Ok this element has an ice front, get indices of the 2 vertices*/
            int seg1[2] = {index[n*3+0]-1,index[n*3+1]-1};
            int seg2[2] = {index[n*3+1]-1,index[n*3+2]-1};
            int seg3[2] = {index[n*3+2]-1,index[n*3+0]-1};
            int pairids[2];
            if (ice_levelset[seg1[0]]>=0 && ice_levelset[seg1[1]]>=0){
                pairids[0] = seg1[0]; pairids[1] = seg1[1];
            }
            else if (ice_levelset[seg2[0]]>=0 && ice_levelset[seg2[1]]>=0){
                pairids[0] = seg2[0]; pairids[1] = seg2[1];
            }
            else if (ice_levelset[seg3[0]]>=0 && ice_levelset[seg3[1]]>=0){
                pairids[0] = seg3[0]; pairids[1] = seg3[1];
            }
            else{
                std::cerr<<"case not supported";
            }
            /*Get normal*/
            ftype len = Kokkos::sqrt(Kokkos::pow(x[pairids[1]]-x[pairids[0]],2) + Kokkos::pow(y[pairids[1]]-y[pairids[0]],2) );
            ftype nx  = +(y[pairids[1]]-y[pairids[0]])/len;
            ftype ny  = -(x[pairids[1]]-x[pairids[0]])/len;
            /*RHS*/
            for(int gg=0;gg<2;gg++){
                ftype phi1 = (1.0 -xg3[gg])/2.;
                ftype phi2 = (1.0 +xg3[gg])/2.;
                ftype bg = base[pairids[0]]*phi1 + base[pairids[1]]*phi2;
                ftype Hg = H[pairids[0]]*phi1 + H[pairids[1]]*phi2;
                bg = Kokkos::min(bg,0.0);
                Fvx[pairids[0]] = Fvx[pairids[0]] +wgt3[gg]/2*1/2*(-rho_w*g* Kokkos::pow(bg,2)+rho*g*Kokkos::pow(Hg,2))*nx*len*phi1;
                Fvx[pairids[1]] = Fvx[pairids[1]] +wgt3[gg]/2*1/2*(-rho_w*g*Kokkos::pow(bg,2)+rho*g*Kokkos::pow(Hg,2))*nx*len*phi2;
                Fvy[pairids[0]] = Fvy[pairids[0]] +wgt3[gg]/2*1/2*(-rho_w*g*Kokkos::pow(bg,2)+rho*g*Kokkos::pow(Hg,2))*ny*len*phi1;
                Fvy[pairids[1]] = Fvy[pairids[1]] +wgt3[gg]/2*1/2*(-rho_w*g*Kokkos::pow(bg,2)+rho*g*Kokkos::pow(Hg,2))*ny*len*phi2;
            } 
        }

        /*One more thing in this element loop: prepare groundedarea needed later for the calculation of basal friction*/
        level[0] = ocean_levelset[index[n*3+0]-1];
        level[1] = ocean_levelset[index[n*3+1]-1];
        level[2] = ocean_levelset[index[n*3+2]-1];
        if (level[0]>=0. && level[1]>=0. && level[2]>=0.){
            /*Completely grounded*/
            groundedratio[n]=1.;
        }
        else if (level[0]<=0. && level[1]<=0. && level[2]<=0.){
            /*Completely floating*/
            groundedratio[n]=0.;
        }
        else{
            /*Partially floating,*/
            ftype s1,s2;
            if (level[0]*level[1]>0){/*Nodes 0 and 1 are similar, so points must be found on segment 0-2 and 1-2*/
                s1=level[2]/(level[2]-level[1]);
                s2=level[2]/(level[2]-level[0]);
            }
            else if (level[1]*level[2]>0){ /*Nodes 1 and 2 are similar, so points must be found on segment 0-1 and 0-2*/
                s1=level[0]/(level[0]-level[1]);
                s2=level[0]/(level[0]-level[2]);
            }
            else if (level[0]*level[2]>0){/*Nodes 0 and 2 are similar, so points must be found on segment 1-0 and 1-2*/
                s1=level[1]/(level[1]-level[0]);
                s2=level[1]/(level[1]-level[2]);
            }
            else{
                std::cerr<<"should not be here...";
            }

            if (level[0]*level[1]*level[2]>0.){
                /*two nodes floating, inner triangle is grounded*/
                groundedratio[n]= s1*s2;
            }
            else{
                /*one node floating, inner triangle is floating*/
                groundedratio[n]= (1.-s1*s2);
            }
        }
    }

    /*Finally add calculation of friction coefficient*/
    ftype* alpha2 = new ftype[nbv];
    for(int i=0;i<nbv;i++){
        /*Compute effective pressure*/
        ftype p_ice   = g*rho*H[i];
        ftype p_water = -rho_w*g*base[i];
        ftype Neff    = p_ice - p_water;
        if (Neff<0.) Neff=0.;
        /*Compute alpha2*/
        alpha2[i] = Kokkos::pow(friction[i],2)*Neff;
    }

    //prepare head and next vectors for chain algorithm, at this point we have not seen any of the elements, so just set the head to -1 (=stop)
    int* head = new int[nbv];
    int* next  = new int[3*nbe];
    for(int i=0;i<nbv;i++) head[i] = -1;

    //Now construct the chain
    for(int k=0;k<nbe;k++){
        for(int j=0;j<3;j++){
            int i;
            int p = 3*k+j;       //unique linear index of current vertex in index
            i = index[p];
            next[p] = head[i - 1];
            head[i -1] = p + 1;
        }
    }

    //Note: Index array starts at 0, but the node# starts at 1
    //Now we can construct the connectivity matrix
    int MAXCONNECT = 8;
    int* connectivity = new int[nbv*MAXCONNECT];
    int* columns = new int[nbv*MAXCONNECT];

    for(int i=0;i<nbv;i++){

        /*Go over all of the elements connected to node I*/
        int count = 0;
        int p=head[i];

        //for (int p = head[i]; p != -1; p = next[p]){
        while (p!= -1){

            int k = p / 3 + 1;     //”row" in index
            int j = (p % 3) - 1;   //"column" in index

            if (j==-1){
                j=2;
                k= k -1;}

            //sanity check
            if (index[p-1] !=i+1){
                std::cout << "Error occurred"  << std::endl;;
            }

            //enter element in connectivity matrix
            connectivity[i * MAXCONNECT + count] = k;
            columns[i * MAXCONNECT + count] = j;
            count++;
            p = next[p-1];
        }
    }

    ftype* device_maxvalx = new ftype[GRID_Xv];
    ftype* device_maxvaly = new ftype[GRID_Xv];
    for(int i=0;i<GRID_Xv;i++) device_maxvalx[i] = 0.;
    for(int i=0;i<GRID_Xv;i++) device_maxvaly[i] = 0.;

    ftype* rho_ML = new ftype[nbv];
    for (int i = 0;i < nbv ;i++){
	    rho_ML[i] = rho*Kokkos::max(60.0,H[i])*ML[i];
    }

    ftype* gr_a_alpha2 = new ftype[nbe*3];
    for(int i = 0;i < nbe; i++){
         if (groundedratio[i] > 0.){
            int n3 = i * 3;
            ftype tmp_gr_a = groundedratio[i] * areas[i];

            for (int j = 0; j < 3; j++){
                 int i_index = index[n3 + j] - 1;
                 gr_a_alpha2[n3 + j] =  tmp_gr_a * alpha2[i_index];
            }
         }
    }

    /*------------Copy all relevant vectors from host (CPU) to device (GPU) ---------------*/
vector(index,int,nbe*3);
vector(rho_ML,ftype,nbv);
vector(gr_a_alpha2,ftype,3 * nbe);
vector(vx,ftype,nbv);
vector(vy,ftype,nbv);
vector(alpha,ftype,3 * nbe);
vector(beta,ftype,nbe*3);
vector(etan,ftype,nbe);
vector(rheology_B,ftype,nbe);
vector(Helem,ftype,nbe);
vector(areas,ftype,nbe);
vector(weights,ftype,nbv);
vector(ML,ftype,nbv);
vector(Fvx,ftype,nbv);
vector(Fvy,ftype,nbv);
vector(dVxdt,ftype,nbv);
vector(dVydt,ftype,nbv);
vector(resolx,ftype,nbv);
vector(resoly,ftype,nbv);
vector(H,ftype,nbv);
vector(spcvx,ftype,nbv);
vector(spcvy,ftype,nbv);
vector(alpha2,ftype,nbv);
vector(groundedratio,ftype,nbe);
vector(isice,bool,nbe);
vector(connectivity,int,nbv*8);
vector(columns,int,nbv*8);
vector(device_maxvalx,ftype,GRID_Xv);
vector(device_maxvaly,ftype,GRID_Xv);

    /*------------ Allocate relevant vectors on device (GPU) host ---------------*/
vector2(KVx,ftype,nbv);
vector2(KVy,ftype,nbv);
vector2(eta_nbv,ftype,nbv);
vector2(Eta_nbe,ftype,nbe*3);
vector2(kvx,ftype,nbe*3);
vector2(kvy,ftype,nbe*3);

    /*Performance metrics*/
    ftype time_s = 0.0;
    ftype mem = (ftype)1e-9*(ftype)nbv*sizeof(ftype);
    int nIO = 8;
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;
    auto t1 = high_resolution_clock::now();

    /*Main loop*/
    std::cout<<"Starting PT loop, nbe="<<nbe<<", nbv="<<nbv<<std::endl; 
    int iter;
    ftype iterror;
    for(iter=1;iter<=niter;iter++){
      
        // iter=11; 

        PT1 (d_vx, d_vy, d_alpha, d_beta, d_index, kvx,  kvy, d_etan, d_Helem, d_areas, d_isice, Eta_nbe, d_rheology_B, n_glen, eta_0, rele, nbe);
    
        PT2_x(kvx, d_groundedratio, d_areas, d_index, d_alpha2, d_vx, d_gr_a_alpha2, d_isice, nbe);
    
        PT2_y (kvy, d_groundedratio, d_areas, d_index, d_alpha2, d_vy, d_gr_a_alpha2, d_isice, nbe);
 
        PT3 (kvx, kvy, Eta_nbe, eta_nbv, d_connectivity, d_columns, d_weights, d_ML, KVx, KVy, d_Fvx, d_Fvy, d_dVxdt, d_dVydt, d_resolx, d_resoly, d_H, d_vx, d_vy, d_spcvx, d_spcvy, d_rho_ML, rho, damp, relaxation, eta_b, nbv);
    
       if ((iter % nout_iter) == 0){
//     //         // //Get final error estimate//
            ftype device_MAXx = 0.0; 
            ftype device_MAXy = 0.0;
            int index_x = 0;
            int index_y = 0;

            Kokkos::deep_copy(h_dVxdt, d_dVxdt);
            Kokkos::deep_copy(h_dVydt, d_dVydt);

            for(int i=0; i<nbv;i++){
                if(h_dVxdt[i]>device_MAXx) index_x=i;
                device_MAXx = std::max(device_MAXx, h_dVxdt[i]);
            }
            for(int i=0; i<nbv;i++){
                if(h_dVydt[i]>device_MAXy) index_y=i;
                device_MAXy = std::max(device_MAXy, h_dVydt[i]);
            }

            if (Kokkos::isnan(device_MAXx) || Kokkos::isnan(device_MAXy)) {break;}
            else {
            iterror = Kokkos::max(device_MAXx, device_MAXy);
            std::cout<<"iter="<<iter<<", err="<<iterror<<std::endl;

            if ((iterror < epsi) && (iter > 100)) break;
       } 
      }  
    }
    //time 
    float seconds = 0;
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    
    seconds = ms_double.count()/1000.0;

    time_s = seconds; ftype gbs = mem/time_s;
    std::cout<<"\n Perf: "<<time_s<<" sec. (@ "<<gbs*(iter-10)*nIO<<" GB/s)"<<std::endl;

    /*Copy results from Device to host*/
    Kokkos::deep_copy(h_vx, d_vx);
    Kokkos::deep_copy(h_vy, d_vy);

    for(int i = 0; i<nbv; i++){vx[i] = h_vx[i];}
    for(int i = 0; i<nbv; i++){vy[i] = h_vy[i];}
    

    /*Write output*/
    fid = fopen(outputfile,"wb");
    if (fid==NULL) std::cerr<<"could not open file " << outputfile << " for binary reading or writing";
    WriteData(fid, "PTsolution", "SolutionType");
    WriteData(fid, vx, nbv, 1, "Vx");
    WriteData(fid, vy, nbv, 1, "Vy");
    if (fclose(fid)!=0) std::cerr<<"could not close file " << outputfile;

    /*Cleanup and return*/
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


    }
    Kokkos::finalize();
    return 0;
}













