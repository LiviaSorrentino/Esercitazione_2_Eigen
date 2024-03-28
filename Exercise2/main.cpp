#include <iostream>
#include "Eigen/Eigen"
#include <iomanip>

using namespace std;
using namespace Eigen;

Vector2d solveLU(const Matrix2d& A, const Vector2d& b)
{
    return A.fullPivLu().solve(b);
}

Vector2d solveQR(const Matrix2d& A, const Vector2d& b)
{
    return A.fullPivHouseholderQr().solve(b);
}

int main()
{
    cout<<scientific;

    //soluzione esatta dei seguenti sistemi è
    VectorXd x(2);
    x<<-1.0e+00, -1.0e+00;

    //Sistema1
    MatrixXd A1(2,2);
    A1<< 5.547001962252291e-01, -3.770900990025203e-02,
        8.320502943378437e-01,-9.992887623566787e-01;

    VectorXd b1(2);
    b1<< -5.169911863249772e-01, 1.672384680188350e-01;

    cout<<"Sistema lineare 1"<<endl;

    Vector2d x1pl = solveLU(A1,b1);
    Vector2d x1qr = solveQR(A1,b1);
    double err1pl = (x1pl-x).norm()/x.norm();
    double err1qr = (x1qr-x).norm()/x.norm();

    cout<< "L'errore relativo associato alla soluzione del primo sistema in fattorizzazione PALU è: "<< err1pl << endl;
    cout<<"L'errore relativo associato alla soluzione del primo sistema in fattorizzazione QR è: "<< err1qr<< endl;

    //Sistema2
    MatrixXd A2(2,2);
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
        8.320502943378437e-01, -8.324762492991313e-01;

    VectorXd b2(2);
    b2<< -6.394645785530173e-04, 4.259549612877223e-04;

    cout<<"Sistema lineare 2"<<endl;

    Vector2d x2pl = solveLU(A2,b2);
    Vector2d x2qr = solveQR(A2,b2);
    double err2pl = (x2pl-x).norm()/x.norm();
    double err2qr = (x2qr-x).norm()/x.norm();

    cout<< "L'errore relativo associato alla soluzione del secondo sistema  in fattorizzazione PALU è: "<< err2pl << endl;
    cout<<"L'errore relativo associato alla soluzione del secondo sistema in fattorizzazione QR è: "<< err2qr<< endl;

    //Sistema3
    MatrixXd A3(2,2);
    A3<< 5.547001962252291e-01, -5.547001955851905e-01,
        8.320502943378437e-01,-8.320502947645361e-01;

    VectorXd b3(2);
    b3<<-6.400391328043042e-10, 4.266924591433963e-10;

    cout<<"Sistema lineare 3"<<endl;

    Vector2d x3pl = solveLU(A3,b3);
    Vector2d x3qr = solveQR(A3,b3);
    double err3pl = (x3pl-x).norm()/x.norm();
    double err3qr = (x3qr-x).norm()/x.norm();

    cout<< "L'errore relativo associato alla soluzione del terzo sistema in fattorizzazione PALU è: "<< err3pl << endl;
    cout<<"L'errore relativo associato alla soluzione del terzo sistema in fattorizzazione QR è: "<< err3qr<< endl;


    return 0;

}
