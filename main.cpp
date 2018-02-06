#include <mpi.h>
#include <iostream>
#include <cmath>
#include <functional>
#include <mkl.h>

const double legendre[4][4] = {
    {0, 0, 0, sqrt(0.5)},
    {0, 0, sqrt(1.5), 0},
    {0, 1.5*sqrt(2.5), 0, -0.5*sqrt(2.5)},
    {2.5*sqrt(3.5), 0, -1.5*sqrt(3.5), 0}
};

double LP(int i, double x) {
    double result = 0;

    for (int k = 0; k < 4; k++) {
        double t = legendre[i][3 - k];

        for (int p = 0; p < k; p++) {
            t *= x;
        }

        result += t;
    }

    return result;
}

double mapX(double x, double a, double b) {
    double p = 2.0 / (b - a);
    double q = - (b + a) / (b - a);

    return p * x + q;
}

double xi(double r, double a, double b) {
    double p = (b-a)*0.5;
    double q = (a+b)*0.5;
    return p*r+q;
}

double quadr(std::function<double (double)> f) {
    double a1 = 2 * sqrt(10.0/7.0);
    double a2 = 13 * sqrt(70);
    //double nodes[3] = { -sqrt(0.6), 0, sqrt(0.6) };
    //double weights[3] = { 5.0/9, 8.0/9, 5.0/9 };

    double nodes[5] = {0, 1.0/3.0*sqrt(5-a1), -1.0/3.0*sqrt(5-a1),  1.0/3.0*sqrt(5+a1), -1.0/3.0*sqrt(5+a1)};
    double weights[5] = {128.0/225, (322+a2)/900.0, (322+a2)/900.0, (322-a2)/900.0, (322-a2)/900.0};

    double result = 0;

    for (int i = 0; i < 5; i++) {
        result += weights[i] * f(nodes[i]);
    }

    return result;
}

class U0 {
public:
    static double expr(double x) {
        if (x>0) {
            return 1.0;
        }
        return 0.0;
        // return 0.5+0.5*sin(M_PI * x);
        //return sin(M_PI*x);
    }
};

void get_left_border(double * C, double * B, int nEl_local, int N, double * u_left) {
    int k = nEl_local;
    double temp = 0;

    for (int i = 0; i < N + 1; i++) {
        temp += C[(k - 1) * (N + 1) + i] * B[2*i + 1];
    }

    u_left[k] = temp; 
}

void get_left_inner(double * C, double * B, int nEl_local, int N, double * u_left) {
    for (int k = 1; k < nEl_local; k++) {
        double temp = 0;
        for (int i = 0; i < N + 1; i++) {
            temp += C[(k - 1) * (N + 1) + i] * B[2*i + 1];
        }
        u_left[k] = temp; 
    }
}

void get_right_border(double * C, double * B, int nEl_local, int N, double * u_right) {
    int k = 0;

    double temp = 0;
    for (int i = 0; i < N + 1; i++) {
        temp += C[k * (N + 1) + i] * B[2*i];
    }
    u_right[k] = temp; 
}

void get_right_inner(double * C, double * B, int nEl_local, int N, double * u_right) {
    for (int k = 1; k < nEl_local; k++) {
        double temp = 0;
        for (int i = 0; i < N + 1; i++) {
            temp += C[k * (N + 1) + i] * B[2*i];
        }
        u_right[k] = temp; 
    }
}

void flux_num(double * u_left, double * u_right, double * flux, int nEl_local) {
    for (int i = 0; i < nEl_local + 1; i++) {
        flux[i] = 0.5*(u_left[i] + u_right[i])+0.5*(u_left[i]-u_right[i]);
    }
}

void get_flux_border(double * u_left, double * u_right, double * B, int nEl_local, int N, double * flux, double * F) {
    flux_num(u_left, u_right, flux, nEl_local);

    int k = 0;
    for (int i = 0; i < N + 1; i++) {
        F[k*(N+1)+i] = flux[k+1] * B[i*2+1] - flux[k] * B[2*i];
    }

    k = nEl_local - 1;
    for (int i = 0; i < N + 1; i++) {
        F[k*(N+1)+i] = flux[k+1] * B[i*2+1] - flux[k] * B[2*i];
    }
}

void get_flux_inner(double * u_left, double * u_right, double * B, int nEl_local, int N, double * flux, double * F) {
    flux_num(u_left, u_right, flux, nEl_local);

    for (int k = 1; k < nEl_local - 1; k++) {
        for (int i = 0; i < N+1; i++) {
            F[k*(N+1)+ i] = flux[k+1] * B[i*2+1] - flux[k] * B[2*i];
        }
    }
}

int main(int argc, char * argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv); 
    
    // TODO pass all this values
   // const double T = atof(argv[1]);
    const int Kt = atof(argv[1]);
    const double dt = atof(argv[2]);
    const int nEl = atof(argv[3]);
    const int N = atof(argv[4]);
    const int NN = 4;
    const double xmin = -1;
    const double xmax = 1;

    const double h = (xmax - xmin) / nEl;

    const double J = h / 2.0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    const int nEl_local = nEl / size;

    double * C = new double[nEl_local * (N + 1)]();
    double * B = new double[(N+1) * 2]();
    double * F = new double[nEl_local * (N+1)]();
    double * u_left = new double[nEl_local + 1]();
    double * u_right = new double[nEl_local + 1]();
    double * flux = new double[nEl_local + 1]();
    double * S = new double[NN * NN]();

    S[NN] = 1.7320508075688772;
    S[2*NN + 1] = 3.8729833462074166;
    S[3*NN] = 2.6457513110645907;
    S[3*NN + 2] = 5.916079783099617;

    double * x_local = new double[nEl_local + 1];

    for (int i = 0; i < nEl_local + 1; i++) {
        x_local[i] = xmin + h * (rank * nEl_local + i);
        //    std::cout << rank << " : " << x_local[i] << std::endl; 
    }

    for (int k = 0; k < nEl_local; k++) {
        for (int i = 0; i < N + 1; i++) {
            auto lambda = [&](double r) { return U0::expr(xi(r, x_local[k], x_local[k + 1])) * LP(i, r); };
            C[k*(N+1)+ i] = quadr(lambda);
        }
    }

    int K = 1000;
    double norm=0;//=new double[Kt+1];
    double norm0 = 0;
    double max_n, tn0;
    int hh = h / K;

      for (int k = 0; k < nEl_local; k++) {
        for (int kk = 0; kk < K; kk++) {
    	    double temp = 0;
    	    double xx = x_local[k] + hh * kk;
    	    for (int i = 0; i < N + 1; i++) {
    		    temp += C[k*(N+1) +i] * LP(i, mapX(xx,x_local[k],x_local[k+1]));
    	    }
    	    double u0 = U0::expr(xx);
    	    norm += (u0 - temp) * (u0 - temp);
    	    norm0 += u0 * u0;
        }
      }
//
//      MPI_Reduce(&norm0, &tn0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
//      MPI_Reduce(&norm, &max_n, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
//
//      if (rank == 0){
//          std::cout<<norm<<" "<<norm0<<std::endl;
//      }


    for (int i = 0; i < N + 1; i++) {
	    B[2*i] = LP(i, -1);
	    B[2*i+1] = LP(i, 1);
    }

    double t = dt;
//double r[N+1];
    double *r = new double[N+1]();
    double *k1 = new double[N+1]();
    double *k2 = new double[N+1]();
    double *k3 = new double[N+1]();
    double *k4 = new double[N+1]();

    MPI_Request request_send_left, request_send_right, request_recv_left, request_recv_right;
    auto f = [&](double * r, double * k) {
        //(S*a-F.col(k))/J;

        // y = alpha * A * x + beta * y
        cblas_dgemv(
            CblasRowMajor,
            CblasNoTrans,
            N+1, N+1, // rows x columns
            1.0/J, // alpha
            S, // A
            NN, // lda
            r, // x
            1, // increment x 
            -1.0/J, // beta
            k, // y
            1 // increment y
            );
    };

    double calc_time = 0;
    double start_time;
    double mpi_time = 0;
    double wait_time = 0;
    double total_time = MPI_Wtime();
    int k;
    int flag;

    int kt = 0;


    for (int kt = 1; kt <= Kt; kt++) {
	// Compute fluxes in border elements
        // Left
	start_time = MPI_Wtime();
        get_left_border(C, B, nEl_local, N, u_left);
        calc_time += MPI_Wtime() - start_time;
               
	// Non blocking send
        start_time = MPI_Wtime();
        MPI_Isend(u_left+nEl_local, 1, MPI_DOUBLE, (rank + 1) % size, 0, MPI_COMM_WORLD, &request_send_right);
        MPI_Irecv(u_left, 1, MPI_DOUBLE, (rank - 1 + size) % size, 0, MPI_COMM_WORLD, &request_recv_left);
      
        mpi_time += MPI_Wtime() - start_time;

	// Right
        start_time = MPI_Wtime();
        get_right_border(C, B, nEl_local, N, u_right);
        calc_time += MPI_Wtime() - start_time;

	// Non blocking send
        start_time = MPI_Wtime();
        MPI_Isend(u_right, 1, MPI_DOUBLE, (rank - 1 + size) % size, 1, MPI_COMM_WORLD, &request_send_left);
        MPI_Irecv(u_right+nEl_local, 1, MPI_DOUBLE, (rank + 1) % size, 1, MPI_COMM_WORLD, &request_recv_right);
        mpi_time += MPI_Wtime() - start_time;
                         
	// Compute inner fluxes
        start_time = MPI_Wtime();
        get_right_inner(C, B, nEl_local, N, u_right);
        get_left_inner(C, B, nEl_local, N, u_left);
        get_flux_inner(u_left, u_right, B, nEl_local, N, flux, F);
	    
        for (k = 1; k < nEl_local - 1; k++) {
            if (k % 100 == 0) {
                MPI_Test(&request_send_left, &flag, MPI_STATUS_IGNORE);
            }

            // r = C.row(k)
            cblas_dcopy(N+1, C + k * (N+1), 1, r, 1);
            
	    // k1 = F.row(k)
            cblas_dcopy(N+1, F + k * (N+1), 1, k1, 1);
            
            f(r,k1);
            
            // r = r + 0.5*dt*k1
            cblas_daxpy(N + 1, dt * 0.5, k1, 1, r, 1);
            // k2 = F.row(k)
            cblas_dcopy(N+1, F + k * (N+1), 1, k2, 1);
            f(r,k2);

            // r = C.row(k)
            cblas_dcopy(N+1, C + k * (N+1), 1, r, 1);
            
	    // r = r+dt*0.5*k2;
            cblas_daxpy(N+1, dt * 0.5, k2, 1, r, 1);
            cblas_dcopy(N+1, F + k * (N+1), 1, k3, 1);
            f(r,k3);
            
            //v4 = r+dt*k3;
            cblas_dcopy(N+1, C + k * (N+1), 1, r, 1);
            cblas_daxpy(N+1, dt, k3, 1, r, 1);
            cblas_dcopy(N+1, F + k * (N+1), 1, k4, 1);
            f(r,k4);
		
            cblas_daxpy(N + 1, dt/6, k1, 1, C + k * (N+1), 1);
            cblas_daxpy(N + 1, dt/3, k2, 1, C + k * (N+1), 1);
            cblas_daxpy(N + 1, dt/3, k3, 1, C + k * (N+1), 1);
            cblas_daxpy(N + 1, dt/6, k4, 1, C + k * (N+1), 1);
        }

        calc_time += MPI_Wtime() - start_time;
               
        start_time = MPI_Wtime();
        MPI_Wait(&request_recv_right, MPI_STATUS_IGNORE);
        MPI_Wait(&request_recv_left, MPI_STATUS_IGNORE);
        wait_time += MPI_Wtime() - start_time;
           
        start_time = MPI_Wtime();
        get_flux_border(u_left, u_right, B, nEl_local, N, flux, F);

        k = 0;
        cblas_dcopy(N+1, C + k * (N+1), 1, r, 1);
        cblas_dcopy(N+1, F + k * (N+1), 1, k1, 1);
        f(r,k1);

        // r = r + 0.5*dt*k1
        cblas_daxpy(N + 1, dt * 0.5, k1, 1, r, 1);
        cblas_dcopy(N+1, F + k * (N+1), 1, k2, 1);
        f(r,k2);
        //   r = r+dt*0.5*k2;
        cblas_dcopy(N+1, C + k * (N+1), 1, r, 1);
        cblas_daxpy(N + 1, dt * 0.5, k2, 1, r, 1);
        cblas_dcopy(N+1, F + k * (N+1), 1, k3, 1);
	f(r,k3);
	//v4 = r+dt*k3;
	cblas_dcopy(N+1, C + k * (N+1), 1, r, 1);
	cblas_daxpy(N + 1, dt, k3, 1, r, 1);
	cblas_dcopy(N+1, F + k * (N+1), 1, k4, 1);
	f(r,k4);

	cblas_daxpy(N + 1, dt/6, k1, 1, C + k * (N+1), 1);
	cblas_daxpy(N + 1, dt/3, k2, 1, C + k * (N+1), 1);
	cblas_daxpy(N + 1, dt/3, k3, 1, C + k * (N+1), 1);
	cblas_daxpy(N + 1, dt/6, k4, 1, C + k * (N+1), 1);

	if (nEl_local > 1) {
		k = nEl_local - 1;

		cblas_dcopy(N+1, C + k * (N+1), 1, r, 1);
		cblas_dcopy(N+1, F + k * (N+1), 1, k1, 1);
		f(r,k1);

		// r = r + 0.5*dt*k1
		cblas_daxpy(N + 1, dt * 0.5, k1, 1, r, 1);
		cblas_dcopy(N+1, F + k * (N+1), 1, k2, 1);
		f(r,k2);
    //   r = r+dt*0.5*k2;
    cblas_dcopy(N+1, C + k * (N+1), 1, r, 1);
    cblas_daxpy(N + 1, dt * 0.5, k2, 1, r, 1);
    cblas_dcopy(N+1, F + k * (N+1), 1, k3, 1);
    f(r,k3);
    //v4 = r+dt*k3;
    cblas_dcopy(N+1, C + k * (N+1), 1, r, 1);
    cblas_daxpy(N + 1, dt, k3, 1, r, 1);
    cblas_dcopy(N+1, F + k * (N+1), 1, k4, 1);
    f(r,k4);

    cblas_daxpy(N + 1, dt/6, k1, 1, C + k * (N+1), 1);
    cblas_daxpy(N + 1, dt/3, k2, 1, C + k * (N+1), 1);
    cblas_daxpy(N + 1, dt/3, k3, 1, C + k * (N+1), 1);
    cblas_daxpy(N + 1, dt/6, k4, 1, C + k * (N+1), 1);
  }

  calc_time += MPI_Wtime() - start_time;

  start_time = MPI_Wtime();
  MPI_Wait(&request_send_left, MPI_STATUS_IGNORE);
  MPI_Wait(&request_send_right, MPI_STATUS_IGNORE);
  wait_time += MPI_Wtime() - start_time;

  //norm[kt] = 0;

 
  //        std::cout <<rank<< " " << norm[kt] << "\t" << norm0[kt] << "\t" << norm[kt] / norm0[kt] << std::endl;

  t += dt;
    }

    //total_time = MPI_Wtime() - total_time;
    total_time = mpi_time + calc_time;

    double max_total;
    double max_norm;
    double local_norm = 0;

    for (int k = 0; k < nEl_local; k++) {
        local_norm=0;
        for (int kk = 0; kk < K; kk++) {
            double temp = 0;
            double xx = x_local[k] + hh * kk;
            for (int i = 0; i < N + 1; i++) {
                temp += C[k*(N+1) +i] * LP(i, mapX(xx,x_local[k],x_local[k+1]));
            }
            double u0 = U0::expr(xx-t);
            local_norm += (u0 - temp) * (u0 - temp);
            norm0 += u0 * u0;
        }
      
        norm += local_norm;
    }
  
  //  int index = cblas_idamax(Kt+1, norm, 1); 
  //  std::cout <<rank << "\t" << index<<" "<<Kt<<" " <<norm[index] << std::endl;
    //       if (int(t / dt) % 100 == 0) {
    double total_norm_0;
    MPI_Reduce(&total_time, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&norm0, &total_norm_0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&norm, &max_norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        //std::cout << "time : " << max_total << std::endl;
        //std::cout << "norm : " << max_norm/(nEl*K) << std::endl;
        std::cout << sqrt(max_norm/total_norm_0) << std::endl;
        //std::cout << max_norm <<" "<< total_norm_0 << std::endl;
    }
    //std::cout <<  " mpi time : " << mpi_time << " total : " << total_time << std::endl;
    //}


    delete [] x_local;
    delete [] u_left;
    delete [] u_right;
    delete [] C;
    delete [] F;
    delete [] S;
   // delete [] r;
    delete [] k1;
    delete [] k2;
    delete [] k3;
    delete [] k4;
    delete [] flux;
//    delete [] norm;

    MPI_Finalize();
}
