module ModuleCommonData
implicit none

integer, parameter :: nstep = 100  ! number of mass points
double precision, parameter :: pi=3.1415926535897932d0

double precision :: redshift       ! redshift
double precision :: omega_0        ! total matter content
double precision :: omega_bar      ! baryon fraction
double precision :: h              ! Hubble constant/100
double precision :: sigma_8        ! variance of linear density field
double precision :: n              ! power spectrum index
double precision :: w_0            ! for dark energy equation of state:
double precision :: w_a            ! w = w_0 + w_a*(1-a)
double precision :: delta_c        ! linear overdensity at virialization
double precision :: Delta          ! overdensity; used only for Tinker MF
double precision :: M_min          ! starting mass
double precision :: M_max          ! ending mass
double precision :: k_max          ! max k for calculating sigma(k)
double precision :: R              ! radius of spherical top-hat
double precision :: AA             ! normalization of the power spectrum 
character(len=4) :: tf             ! transfer function used
character(len=6) :: fitting_f      ! mass fitting function used

end module ModuleCommonData
