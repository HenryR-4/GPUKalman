#pragma once

#include <type_traits>

#include "Matrix.h"

template<typename T>
class KalmanFilterState {
    public:
        KalmanFilterState<T>(const KalmanFilterState<T>& other)
        {
            this->x = other.x;
            this->P = other.P;
        }

        KalmanFilterState<T>(const DeviceMatrix<T>& x, const DeviceMatrix<T>& P) 
        {
            this->x = x;
            this->P = P;
        }

        DeviceMatrix<float> x;
        DeviceMatrix<float> P;
};

template<typename T>
class KalmanFilter
{
    static_assert(std::is_same<T,float>::value, "Only 'float' supported currently.");
    public:
        KalmanFilter(
            const DeviceMatrix<T>& F,
            const DeviceMatrix<T>& Q,
            const DeviceMatrix<T>& R,
            const DeviceMatrix<T>& H
        );

        KalmanFilter(
            DeviceMatrix<T>&& F,
            DeviceMatrix<T>&& Q,
            DeviceMatrix<T>&& R,
            DeviceMatrix<T>&& H
        );

        void run(const DeviceMatrix<T>& z, KalmanFilterState<T>& state);

    private:
        DeviceMatrix<T> F_;
        DeviceMatrix<T> Q_;
        DeviceMatrix<T> R_;
        DeviceMatrix<T> H_;
        DeviceMatrix<T> I_;

        void predict(KalmanFilterState<T>& state);
        void update(const DeviceMatrix<T>& z, KalmanFilterState<T>& state);
};

template<typename T>
KalmanFilter<T>::KalmanFilter(
    const DeviceMatrix<T>& F,
    const DeviceMatrix<T>& Q,
    const DeviceMatrix<T>& R,
    const DeviceMatrix<T>& H
) : F_(F),
    Q_(Q),
    R_(R),
    H_(H)
{
    HostMatrix<T> I = HostMatrix<T>(H_.width(), H_.width());
    for (size_t i = 0; i < I.height(); i++) {
        for (size_t j = 0; j < I.width(); j++) {
            I(i,j) = (i==j) ? 1 : 0;
        }
    }

    I_ = DeviceMatrix<T>(I);
}

template<typename T>
KalmanFilter<T>::KalmanFilter(
    DeviceMatrix<T>&& F,
    DeviceMatrix<T>&& Q,
    DeviceMatrix<T>&& R,
    DeviceMatrix<T>&& H
) : F_(std::move(F)),
    Q_(std::move(Q)),
    R_(std::move(R)),
    H_(std::move(H))
{
    HostMatrix<T> I = HostMatrix<T>(H_.width(), H_.width());
    for (size_t i = 0; i < I.height(); i++) {
        for (size_t j = 0; j < I.width(); j++) {
            I(i,j) = (i==j) ? 1 : 0;
        }
    }

    I_ = DeviceMatrix<T>(I);
}

template<typename T>
void KalmanFilter<T>::run(const DeviceMatrix<T>& z, KalmanFilterState<T>& state)
{
    predict(state);
    update(z, state);
}

template<typename T>
void KalmanFilter<T>::predict(KalmanFilterState<T>& state)
{
    state.x = F_*state.x;
    state.P = abatpc(F_, state.P, Q_);
}

template<typename T>
void KalmanFilter<T>::update(const DeviceMatrix<T>& z, KalmanFilterState<T>& state)
{
    DeviceMatrix<T> K = mul(state.P, false, H_, true)*invert(abatpc(H_, state.P, R_));
    state.x += K*(z - H_*state.x);
    state.P = (I_-K*H_)*state.P;
}
