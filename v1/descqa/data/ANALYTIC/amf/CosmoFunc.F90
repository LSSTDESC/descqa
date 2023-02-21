!-----------------------------------------------------------------
! Collection of transfer functions and mass function fits
! 

! use as: s2 = sigma2(k)     where k is the wavenumber in h/Mpc
!         mf = tinker(sigma) where sigma is the variance of the linear density


module ModuleCosmoFunc

implicit none
public  :: sigma2, press_schechter, sheth_tormen, jenkins, &
           delpopolo, reed, reed06, reed_z, tinker
private :: CMB, BBKS, EBW, PD, HS, KH, EH
double precision, parameter, private :: T_CMB = 2.728d0  ! Fixsen et al. 1996

contains

double precision function CMB(k)
use ModuleCommonData, ONLY: k_max, omega_0, omega_bar
use ModuleInterpolator, ONLY: interpolate
implicit none
double precision, intent(in) :: k
integer :: i
integer, save :: nmax
double precision    :: tmp, tfcdm, tfbar
double precision, dimension(:), allocatable, save :: kk, tanf
logical, save :: first_call = .true.
logical :: fexist

if (first_call) then
   inquire(file='cmb.tf', exist=fexist)
   if (.not. fexist) STOP 'Need cmb.tf file for CMBFAST transfer function!'
   open(15, file='cmb.tf', status='old')
   nmax = 0
   do
      read(15,*,end=10) tmp, tmp, tmp, tmp, tmp, tmp, tmp
      nmax = nmax+1
   enddo
10 rewind(15)
   allocate(kk(nmax), tanf(nmax))
   do i = 1, nmax
      read(15,*) kk(i), tfcdm, tfbar, tmp, tmp, tmp, tmp
      tanf(i) = tfbar*omega_bar/omega_0 + tfcdm*(omega_0-omega_bar)/omega_0
   enddo
   close(15)

   if (k_max .gt. kk(nmax)) then
      k_max = kk(nmax)
      write(*,*) "Reseting k_max to ", k_max
   endif

   tanf(:) = log10(tanf(:) / tanf(1))  ! normalize to T(k=0)=1
   kk(:)   = log10(kk(:))
   first_call = .false.
endif
CMB = 10.0d0**interpolate(nmax, kk, tanf, log10(k))
end function CMB


double precision function BBKS(k)
use ModuleCommonData, ONLY: omega_0, h
implicit none
double precision, intent(in) :: k
double precision :: q
q    = k / (omega_0*h)
BBKS = log(1.0d0+2.34d0*q)/(2.34d0*q) * &
       (1.0d0 + 3.89d0*q + (16.1d0*q)**2 + (5.46d0*q)**3 + (6.71d0*q)**4)**(-0.25d0)
end function BBKS


double precision function EBW(k)
use ModuleCommonData, ONLY: omega_0, h
implicit none
double precision, intent(in) :: k
double precision, save :: gam, a, b, cc, ni
logical, save :: first_call=.true.
if (first_call) then
   gam   = omega_0*h
   a     = 6.4d0/gam
   b     = 3.0d0/gam
   cc    = 1.7d0/gam
   ni    = 1.13d0
   first_call=.false.
endif
EBW = (1.0d0 + (a*k+(b*k)**1.5d0+(cc*k)**2)**ni)**(-1.0d0/ni)
end function EBW


double precision function PD(k)
use ModuleCommonData, ONLY: omega_0, omega_bar, h
implicit none
double precision, intent(in) :: k
double precision :: q
q  = k / (omega_0*h*exp(-2.0d0*omega_bar))
PD = log(1.0d0+2.34d0*q)/(2.34d0*q) * &
     (1.0d0 + 3.89d0*q + (16.1d0*q)**2 + (5.46d0*q)**3 + (6.71d0*q)**4)**(-0.25d0)
end function PD


double precision function HS(k)
use ModuleCommonData, ONLY: omega_0, omega_bar, h
implicit none
double precision, intent(in) :: k
double precision, parameter :: tt = (T_CMB/2.7d0)**2
double precision :: a1, a2, q, omh2
double precision, save :: alpha
logical, save :: first_call=.true.
if (first_call) then
   omh2 = omega_0*h*h
   a1 = (46.9d0*omh2)**0.67d0  * (1.0d0+(32.1d0*omh2)**(-0.532d0))
   a2 = (12.0d0*omh2)**0.424d0 * (1.0d0+(45.0d0*omh2)**(-0.582d0))
   alpha = a1**(-omega_bar/omega_0) * a2**(-(omega_bar/omega_0)**3)
   first_call = .false.
endif
q  = k*tt / (omega_0*h*sqrt(alpha))
HS = log(1.0d0+2.34d0*q)/(2.34d0*q) * &
     (1.0d0 + 3.89d0*q + (16.1d0*q)**2 + (5.46d0*q)**3 + (6.71d0*q)**4)**(-0.25d0)
end function HS


double precision function KH(k)
use ModuleCommonData, ONLY: omega_0, omega_bar, h
implicit none
double precision, intent(in) :: k
double precision, parameter :: tt = (T_CMB/2.7d0)**2
double precision :: a1, a2, q, omh2
double precision, save :: alpha
logical, save :: first_call=.true.
if (first_call) then
   omh2 = omega_0*h*h
   a1 = (46.9d0*omh2)**0.67d0  * (1.0d0+(32.1d0*omh2)**(-0.532d0))
   a2 = (12.0d0*omh2)**0.424d0 * (1.0d0+(45.0d0*omh2)**(-0.582d0))
   alpha = a1**(-omega_bar/omega_0) * a2**(-(omega_bar/omega_0)**3)
   first_call = .false.
endif
q  = k*tt / (omega_0*h*sqrt(alpha)*(1-omega_bar/omega_0)**0.6d0)
KH = log(1.0d0+2.34d0*q)/(2.34d0*q) * &
     (1.0d0 + 13.0d0*q + (10.5d0*q)**2 + (10.4d0*q)**3 + (6.51d0*q)**4)**(-0.25d0)
end function KH


double precision function EH(k)
use ModuleCommonData, ONLY: omega_0, omega_bar, h
implicit none
double precision, intent(in) :: k
double precision, parameter :: omega_nu=0.0d0, N_nu=3.04d0, tt=(T_CMB/2.7d0)**2
double precision, save :: f_nu, alpha_nu, sound_horizon, beta_c
double precision :: f_bar, f_nub, f_c, f_cb, p_c, p_cb
double precision :: omhh, obhh, z_drag, z_equality, y_d, R_drag, R_equality, k_equality
double precision :: kk, q, q_eff, q_nu, gamma_eff
logical, save :: first_call=.true.
if (first_call) then
   f_nu  = omega_nu/omega_0
   f_bar = omega_bar/omega_0
   f_nub = f_nu + f_bar
   f_c  = 1.0d0-f_nu-f_bar
   f_cb = 1.0d0-f_nu
   p_c  = (5.0d0-sqrt(1.0d0+24.0d0*f_c)) / 4.0d0
   p_cb = (5.0d0-sqrt(1.0d0+24.0d0*f_cb)) / 4.0d0

   omhh = omega_0*h*h
   obhh = omega_bar*h*h
   z_equality = 2.50d4*omhh/tt**2
   z_drag = 0.313d0/omhh**0.419d0 * (1.0d0+0.607d0*omhh**0.674d0)
   z_drag = 1.0d0 + z_drag*obhh**(0.238d0*omhh**0.223d0);
   z_drag = 1291.0d0 * omhh**0.251d0 / (1.0d0 + 0.659d0*omhh**0.828d0) * z_drag
   y_d = (1.0d0+z_equality) / (1.0d0+z_drag)

   alpha_nu = (f_c/f_cb) * (5.d0-2.d0*(p_c+p_cb))/(5.d0-4.d0*p_cb)
   alpha_nu = alpha_nu*(1.d0-0.553d0*f_nub+0.126d0*f_nub**3)
   alpha_nu = alpha_nu/(1.d0-0.193d0*sqrt(f_nu*N_nu)+0.169d0*f_nu*N_nu**0.2d0)
   alpha_nu = alpha_nu*(1.d0+y_d)**(p_cb-p_c)
   alpha_nu = alpha_nu*(1.d0+0.5d0*(p_c-p_cb)*(1.d0+1.d0/(3.d0-4.d0*p_c)/(7.d0-4.d0*p_cb)) / &
                        (1.d0+y_d))

   k_equality = 0.0746d0*omhh/tt
   R_drag = 31.5d0*obhh/tt**2 * 1.0d3/(1.d0+z_drag)
   R_equality = 31.5d0*obhh/tt**2 * 1.0d3/(1.d0+z_equality)
   sound_horizon = 2.d0/3.d0/k_equality*sqrt(6.d0/R_equality) * &
                   log(( sqrt(1.d0+R_drag)+sqrt(R_drag+R_equality) )/(1.d0+sqrt(R_equality)))
   beta_c=1.0d0/(1.0d0-0.949d0*f_nub)

   first_call = .false.
endif
kk = k*h
omhh = omega_0*h*h
q = kk*tt/omhh
gamma_eff = omhh*(sqrt(alpha_nu) + (1.d0-sqrt(alpha_nu)) / &
                                   (1.d0+(0.43d0*kk*sound_horizon)**4))
q_eff = kk*tt/gamma_eff
EH = log(exp(1.d0)+1.84d0*beta_c*sqrt(alpha_nu)*q_eff)
EH = EH/(EH + q_eff*q_eff*(14.4d0 + 325.d0/(1.d0+60.5d0*q_eff**1.11d0)))
q_nu = 3.92d0*q*sqrt(N_nu)/f_nu;
EH = EH*(1.d0+(1.2d0*f_nu**0.64d0*N_nu**(0.3d0+0.6d0*f_nu)) / &
              (q_nu**(-1.6d0)+q_nu**0.8d0))
end function EH


double precision function sigma2(k)
use ModuleCommonData, ONLY: R, AA, n, tf, pi
implicit none
double precision, intent(in) :: k
double precision, parameter :: inv2pi = 1.d0/(2.d0*pi*pi)
double precision :: window_f, trf
if (k .le. tiny(1.d0)) then
   sigma2 = 0.d0
   return
endif
select case(tf)
   case("CMB")
      trf = CMB(k)
   case("BBKS")
      trf = BBKS(k)
   case("EBW")
      trf = EBW(k)
   case("PD")
      trf = PD(k)
   case("HS")
      trf = HS(k)
   case("KH")
      trf = KH(k)
   case("EH")
      trf = EH(k)
   case default
      STOP 'transfer function not supported'
end select
window_f = 3.d0*((sin(k*R) - k*R*cos(k*R))/((k*R)**3))
sigma2   = inv2pi * k*k * AA*k**n * trf*trf * window_f*window_f
end function sigma2


! Mass function fits:

double precision function press_schechter(sigma)
use ModuleCommonData, ONLY: pi, delta_c
implicit none
double precision, intent(in) :: sigma
press_schechter = sqrt(2.d0/pi)*delta_c/sigma*exp(-delta_c**2/(2.d0*sigma**2))
end function press_schechter


double precision function sheth_tormen(sigma)
use ModuleCommonData, ONLY: pi, delta_c
implicit none
double precision, intent(in) :: sigma
double precision, parameter :: a=0.707d0, p=0.3d0
sheth_tormen = 0.3222d0*sqrt(2.d0*a/pi)*(1.d0+(sigma**2/(a*delta_c**2))**p) * &
               delta_c/sigma * exp(-a*delta_c**2/(2.d0*sigma**2))
end function sheth_tormen


double precision function jenkins(sigma)
implicit none
double precision, intent(in) :: sigma
jenkins = 0.315d0 * exp(-(abs(log(1.d0/sigma) + 0.61d0)**3.8d0))
end function jenkins


double precision function lanl(sigma)
implicit none
double precision, intent(in) :: sigma
lanl = 0.7234d0 * (sigma**(-1.625d0)+0.2538d0) * exp(-1.1982d0/sigma**2)
end function lanl


double precision function delpopolo(sigma)
use ModuleCommonData, ONLY: pi, delta_c
implicit none
double precision, intent(in) :: sigma
double precision, parameter :: a=0.707d0, A5=1.75d0
double precision :: ni
ni = (delta_c/sigma)**2
delpopolo = A5*(1.d0+0.1218d0/((a*ni)**0.585d0))*sqrt(a*ni/(2.d0*pi))* &
               exp(-0.4019d0*a*ni*(1.d0+0.5526d0/((a*ni)**0.585d0)+0.02d0/ &
                                             ((a*ni)**0.4d0) )**2)
end function delpopolo


double precision function reed(sigma)
implicit none
double precision, intent(in) :: sigma
reed = sheth_tormen(sigma) * exp(-0.7d0/(sigma*(cosh(2.d0*sigma))**5))
end function reed


double precision function reed06(sigma)
use ModuleCommonData, ONLY: pi, delta_c
implicit none
double precision, intent(in) :: sigma
double precision :: nu, nu_prime, lnsigmainv, lngauss1
double precision, parameter :: sqrt_two_over_pi=sqrt(2.d0/pi)
nu         = delta_c/sigma
nu_prime   = sqrt(0.707d0)*nu
lnsigmainv = log(1.0d0/sigma)
lngauss1   = exp(-(lnsigmainv-0.4d0)**2/(2.d0*0.6d0**2))
reed06     = 0.3222d0*sqrt_two_over_pi*nu_prime*exp(-1.08d0*nu_prime**2/2.d0) * &
             (1.d0 + 1.d0/nu_prime**0.6d0 + 0.2d0*lngauss1)
end function reed06


double precision function reed_z(sigma, n_eff)
use ModuleCommonData, ONLY: pi, delta_c
implicit none
double precision, intent(in) :: sigma, n_eff
double precision :: nu, nu_prime, lnsigmainv, lngauss1, lngauss2
double precision, parameter :: sqrt_two_over_pi=sqrt(2.d0/pi)
nu         = delta_c/sigma
nu_prime   = sqrt(0.707d0)*nu
lnsigmainv = log(1.d0/sigma)
lngauss1   = exp(-(lnsigmainv-0.4d0 )**2 / (2.d0*0.6d0**2))
lngauss2   = exp(-(lnsigmainv-0.75d0)**2 / (2.d0*0.2d0**2))
reed_z     = 0.3222d0 * sqrt_two_over_pi * nu_prime * &
             (1.d0 + 1.d0/nu_prime**0.6d0 + 0.6d0*lngauss1 + 0.4d0*lngauss2) * &
             exp(-1.08d0*nu_prime**2/2.d0 - 3.d-2/((n_eff+3.d0)**2)*nu**0.6d0)
end function reed_z


double precision function tinker(sigma)
use ModuleCommonData, ONLY: redshift, Delta
implicit none
double precision, intent(in) :: sigma
double precision :: alpha, An, a, b, c
alpha = 10.0d0**(-(0.75d0/log10(Delta/75.0d0))**1.2d0)
An = (0.1d0 * log10(Delta) - 0.05d0) * (1.0+redshift)**(-0.26d0)
a  = (1.43d0 + (log10(Delta) - 2.3d0)**1.5d0) * (1.d0+redshift)**(-0.06d0)
b  = (1.d0 + (log10(Delta) - 1.6d0)**(-1.5d0)) * (1.d0+redshift)**(-alpha)
c  = 1.2d0 + abs((log10(Delta) - 2.35d0))**1.6d0
tinker = An*((sigma/b)**(-a) + 1.0d0)*exp(-c/sigma**2)
end function tinker

end module ModuleCosmoFunc
