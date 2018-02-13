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
        return sin(M_PI*x);
//        if (x>0) {
//            return 1.0;
//        }
//        return 0.0;
        // return 0.5+0.5*sin(M_PI * x);
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

struct Params {
    double dt;
    double h;
    int N;
    int nEl_local;;
};

class Solver {
    // used degree 
    int N;

    // max number of basis functions
    static const int NN = 4;
    double * S, *k1, *k2, *k3, *k4, *r, *C, *F, *x_local;

    const double J;
    Params * p;
    // k = S*r - 1/J*k
    void f (double * r, double * k) {

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

public:
    Solver(double * aC, double * aF, double * ax_local, Params & ap): C(aC), F(aF), x_local(ax_local), p(&ap), J(ap.h * 0.5) {
        k1 = new double[p->N+1]();
        k2 = new double[p->N+1]();
        k3 = new double[p->N+1]();
        k4 = new double[p->N+1]();
        S = new double[NN * NN]();
        r = new double[N+1]();
 
        // Stiffness matrix
        S[4] = 1.7320508075688772;
        S[2*4 + 1] = 3.8729833462074166;
        S[3*4] = 2.6457513110645907;
        S[3*4 + 2] = 5.916079783099617;
    }
    ~Solver() {
        delete [] r;
        delete [] k1;
        delete [] k2;
        delete [] k3;
        delete [] k4;
        delete [] S;
    }

    void compute_coefficients(int k) {
        // r = C.row(k)
        cblas_dcopy(p->N+1, C + k * (p->N+1), 1, r, 1);
        // k1 = F.row(k)
        cblas_dcopy(p->N+1, F + k * (p->N+1), 1, k1, 1);

        f(r,k1);

        // r = r + 0.5*p->dt*k1
        cblas_daxpy(p->N + 1, p->dt * 0.5, k1, 1, r, 1);
        // k2 = F.row(k)
        cblas_dcopy(p->N+1, F + k * (p->N+1), 1, k2, 1);
        f(r,k2);

        // r = C.row(k)
        cblas_dcopy(p->N+1, C + k * (p->N+1), 1, r, 1);
        // r = r+p->dt*0.5*k2;
        cblas_daxpy(p->N+1, p->dt * 0.5, k2, 1, r, 1);
        cblas_dcopy(p->N+1, F + k * (p->N+1), 1, k3, 1);
        f(r,k3);

        //v4 = r+p->dt*k3;
        cblas_dcopy(p->N+1, C + k * (p->N+1), 1, r, 1);
        cblas_daxpy(p->N+1, p->dt, k3, 1, r, 1);
        cblas_dcopy(p->N+1, F + k * (p->N+1), 1, k4, 1);
        f(r,k4);

        cblas_daxpy(p->N + 1, p->dt/6, k1, 1, C + k * (p->N+1), 1);
        cblas_daxpy(p->N + 1, p->dt/3, k2, 1, C + k * (p->N+1), 1);
        cblas_daxpy(p->N + 1, p->dt/3, k3, 1, C + k * (p->N+1), 1);
        cblas_daxpy(p->N + 1, p->dt/6, k4, 1, C + k * (p->N+1), 1);   
    }

    void compute_norm(double & norm, double t) {
        double element_norm;
        double local_norm = 0;
        int K = 1000;
        double hh = p->h / K;

        for (int k = 0; k < p->nEl_local; k++) {
            local_norm=0;
            for (int kk = 0; kk < K; kk++) {
                double temp = 0;
                double xx = x_local[k] + hh * kk;
                for (int i = 0; i < p->N + 1; i++) {
                    temp += C[k*(N+1) +i] * LP(i, mapX(xx,x_local[k],x_local[k+1]));
                }
                double u0 = U0::expr(xx-t);
                element_norm += (u0 - temp) * (u0 - temp) * hh * hh;
            }

            local_norm += local_norm;
        }

        MPI_Reduce(&local_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
};


int main(int argc, char * argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv); 

    // TODO pass all this values
   // const double T = atof(argv[1]);
    const int Kt = atof(argv[1]);
    const double dt = atof(argv[2]);
    const int nEl = atof(argv[3]);
    const int N = atof(argv[4]);
    const double xmin = -1;
    const double xmax = 1;
    double norm;
    int flag;

    const double h = (xmax - xmin) / nEl;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int nEl_local = nEl / size;
    Params p({dt, h, N, nEl_local});

    double * C = new double[nEl_local * (N + 1)]();
    double * B = new double[(N+1) * 2]();
    double * F = new double[nEl_local * (N+1)]();

    // value from the left
    double * u_left = new double[nEl_local + 1]();
    //value from the right
    double * u_right = new double[nEl_local + 1]();

    // flux value
    double * flux = new double[nEl_local + 1]();

    // element borders
    double * x_local = new double[nEl_local + 1];

    for (int i = 0; i < nEl_local + 1; i++) {
        x_local[i] = xmin + h * (rank * nEl_local + i);
    }

    Solver solver(C, F, x_local, p);

    // Initial conditions
    for (int k = 0; k < nEl_local; k++) {
        for (int i = 0; i < N + 1; i++) {
            auto lambda = [&](double r) { return U0::expr(xi(r, x_local[k], x_local[k + 1])) * LP(i, r); };
            C[k*(N+1)+ i] = quadr(lambda);
        }
    }

    for (int i = 0; i < N + 1; i++) {
        B[2*i] = LP(i, -1);
        B[2*i+1] = LP(i, 1);
    }

    MPI_Request request_send_left, request_send_right, request_recv_left, request_recv_right;

    // metrics
    double calc_time = 0;
    double start_time;
    double mpi_time = 0;
    double wait_time = 0;
    double total_time = MPI_Wtime();

    // pointer to k-th element
    int k;

    int kt = 0;

    double t = dt;
    // Kt iterations
    for (int kt = 1; kt <= Kt; kt++) {
        // Compute left value on the border of a partition
        start_time = MPI_Wtime();
        get_left_border(C, B, nEl_local, N, u_left);
        calc_time += MPI_Wtime() - start_time;

        // Non-blocking send of the computed value to the right
        start_time = MPI_Wtime();
        MPI_Isend(u_left+nEl_local, 1, MPI_DOUBLE, (rank + 1) % size, 0, MPI_COMM_WORLD, &request_send_right);
        MPI_Irecv(u_left, 1, MPI_DOUBLE, (rank - 1 + size) % size, 0, MPI_COMM_WORLD, &request_recv_left);

        mpi_time += MPI_Wtime() - start_time;

        // Compute right value on the border
        start_time = MPI_Wtime();
        get_right_border(C, B, nEl_local, N, u_right);
        calc_time += MPI_Wtime() - start_time;

        // Non-blocking send to the left
        start_time = MPI_Wtime();
        MPI_Isend(u_right, 1, MPI_DOUBLE, (rank - 1 + size) % size, 1, MPI_COMM_WORLD, &request_send_left);
        MPI_Irecv(u_right+nEl_local, 1, MPI_DOUBLE, (rank + 1) % size, 1, MPI_COMM_WORLD, &request_recv_right);
        mpi_time += MPI_Wtime() - start_time;
                         
        // Compute left and right value on inner elements
        start_time = MPI_Wtime();
        get_right_inner(C, B, nEl_local, N, u_right);
        get_left_inner(C, B, nEl_local, N, u_left);

        // Compute inner flux
        get_flux_inner(u_left, u_right, B, nEl_local, N, flux, F);

        // Compute coefficients for each inner element
        for (k = 1; k < nEl_local - 1; k++) {
            // give control back to MPI
            if (k % 100 == 0) {
                MPI_Test(&request_send_left, &flag, MPI_STATUS_IGNORE);
            }
            solver.compute_coefficients(k);
        }

        calc_time += MPI_Wtime() - start_time;

        // Receive border values
        start_time = MPI_Wtime();
        MPI_Wait(&request_recv_right, MPI_STATUS_IGNORE);
        MPI_Wait(&request_recv_left, MPI_STATUS_IGNORE);
        wait_time += MPI_Wtime() - start_time;

        // Compute fluxes on boundaries
        start_time = MPI_Wtime();
        get_flux_border(u_left, u_right, B, nEl_local, N, flux, F);

        // Compute coefficients for the first local element
        k = 0;
        solver.compute_coefficients(k);

        // Compute coefficients for the last local element
        if (nEl_local > 1) {
            k = nEl_local - 1;
            solver.compute_coefficients(k);
        }

        calc_time += MPI_Wtime() - start_time;

        start_time = MPI_Wtime();
        MPI_Wait(&request_send_left, MPI_STATUS_IGNORE);
        MPI_Wait(&request_send_right, MPI_STATUS_IGNORE);
        wait_time += MPI_Wtime() - start_time;

        t += dt;
    }

    //total_time = MPI_Wtime() - total_time;
    total_time = mpi_time + calc_time;

    double max_total;
    MPI_Reduce(&total_time, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    solver.compute_norm(norm, t);

    if (rank == 0) {
        std::cout << sqrt(norm) << std::endl;
    }

    delete [] x_local;
    delete [] u_left;
    delete [] u_right;
    delete [] C;
    delete [] F;
    delete [] flux;

    MPI_Finalize();
}
