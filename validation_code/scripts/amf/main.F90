!------------------------------------------------------------------------
!
! Program AMF (Analytic Mass Function) tabulates mass function values
!             given in many papers as f(sigma).  See README for options.

!             Units are M_sun/h and Mpc/h

!                                                 Zarija Lukic
!                                        Urbana-Champaign, November 2005
!                                             zlukic@astro.uiuc.edu


program amf

use ModuleCommonData 
implicit none

integer :: i
double precision, dimension(nstep,4) :: mf  ! output from the 'main' routine
character(len=10) :: arg, in_string
character(len=1), parameter :: nl=NEW_LINE("A"), tab=char(9)
logical :: fexist

inquire(file='input.par', exist=fexist)
if (fexist) then
   open (10, file='input.par', status='old')
   read (10,*) omega_0
   read (10,*) omega_bar
   read (10,*) h
   read (10,*) sigma_8
   read (10,*) n
   read (10,*) w_0
   read (10,*) w_a
   read (10,*) delta_c
   read (10,*) tf
   read (10,*) fitting_f
   read (10,*) redshift
   read (10,*) Delta
   read (10,*) M_min
   read (10,*) M_max
   read (10,*) k_max
   close(10)
else
   STOP 'Need input.par file !'
endif

do i = 1, iargc(), 2
   call getarg(i, arg)
   select case(arg)
      case("-omega_0", "-Omega_0", "-omega_m", "-Omega_m")
         call getarg(i+1, in_string)
         read (in_string, *) omega_0
      case("-omega_b", "-Omega_b", "-omega_bar", "-Omega_bar")
         call getarg(i+1, in_string)
         read (in_string, *) omega_bar
      case("-h", "-h0", "-h_0")
         call getarg(i+1, in_string)
         read (in_string, *) h
      case("-sigma_8", "-sigma8", "-Sigma_8", "-Sigma8")
         call getarg(i+1, in_string)
         read (in_string, *) sigma_8
      case("-n", "-n_s", "-ns")
         call getarg(i+1, in_string)
         read (in_string, *) n
      case("-w_0", "-w0")
         call getarg(i+1, in_string)
         read (in_string, *) w_0
      case("-w_a", "-wa")
         call getarg(i+1, in_string)
         read (in_string, *) w_a
      case("-delta_c")
         call getarg(i+1, in_string)
         read (in_string, *) delta_c
      case("-tf", "-transfer_function")
         call getarg(i+1, in_string)
         read (in_string, *) tf
      case("-f", "-fitting", "-fitting_f")
         call getarg(i+1, in_string)
         read (in_string, *) fitting_f
      case("-z", "-redshift")
         call getarg(i+1, in_string)
         read (in_string, *) redshift
      case("-delta", "-Delta")
         call getarg(i+1, in_string)
         read (in_string, *) Delta
      case("-M_min", "-Mmin")
         call getarg(i+1, in_string)
         read (in_string, *) M_min
      case("-M_max", "-Mmax")
         call getarg(i+1, in_string)
         read (in_string, *) M_max
      case("-k_max")
         call getarg(i+1, in_string)
         read (in_string, *) k_max
      case default
         write(*,*) "Usage: ./amf.exe [-param xxx]"
         write(*,*) "Full list of parameters:"
         write(*,*) tab, "-omega_0", nl, tab, "-omega_bar", nl, tab, "-h", &
                    nl, tab, "-sigma_8", nl, tab, "-n_s", nl, tab, "-w_0", &
                    nl, tab, "-w_a", nl, tab, "-delta_c", nl, tab, "-tf", &
                    nl, tab, "-fitting", nl, tab, "-z", nl, tab, "-Delta", &
                    nl, tab, "-M_min", nl, tab, "-M_max", nl, tab, "-k_max"
         STOP
   end select
enddo

if (redshift .lt. 0.0d0) STOP 'Wrong redshift is given !'

call main(mf)

open(20, file='analytic.dat', status='unknown')
write(20,*) '#    sigma       f(sigma)          M         dn/dlogM'
write(20,*) '# ------------------------------------------------------'
do i=1, nstep
   write(20, '(4(1X,ES13.6))') mf(i,:)
enddo
close(20)
write(*,*) 'Output is written in analytic.dat'

end program amf


! ----------------------

subroutine main(mf)

use ModuleCommonData
use ModuleCosmoFunc
use ModuleGrowthFactor, ONLY: linear_growth
use ModuleIntegrator, ONLY: integrate
implicit none

double precision, dimension(nstep,4), intent(out) :: mf
integer :: i
double precision :: s2, M, dM, jakobian, rho_0, growth
double precision, dimension(nstep+2,3) :: f ! this is a temporary array

double precision, parameter :: k_min = 0.0d0
double precision, parameter :: rho_c = 3.0d4/(8.0d0*pi*6.673d-11) * 1.551435d-2

! normalize P(k) to the given sigma_8:
R  = 8.0d0
AA = 1.0d0
s2 = integrate(sigma2, k_min, k_max)
AA = AA * sigma_8*sigma_8 / s2

call linear_growth(redshift, growth)
write(*,*) 'z = ', redshift, 'd(z) = ', growth
write(*,*) 'mass function: ', fitting_f

rho_0 = rho_c*omega_0
dM = log10(M_max/M_min) / (nstep-1)
M  = M_min / 10.0d0**(dM)
do i = 1, nstep+2
   R = (3.0d0*M/(4.0d0*pi*rho_0))**(1.0d0/3.0d0)
   s2 = integrate(sigma2, k_min, k_max)
   f(i,1) = M
   f(i,2) = sqrt(s2) * growth
   select case(fitting_f)
      case('PS')
         f(i,3) = press_schechter(f(i,2))
      case('ST')
         f(i,3) = sheth_tormen(f(i,2))
      case('JEN')
         f(i,3) = jenkins(f(i,2))
      case('LANL')
         f(i,3) = lanl(f(i,2))
      case('DELP')
         f(i,3) = delpopolo(f(i,2))
      case('REED')
         f(i,3) = reed(f(i,2))
      case('REED06')
         f(i,3) = reed06(f(i,2))
      case('TINK')
         f(i,3) = tinker(f(i,2))
      case default
         STOP 'fitting function not supported'
   end select
   f(i,3) = max(f(i,3), tiny(1.0d0))
   M = M * 10.0d0**(dM)
enddo

do i = 2, nstep+1
   ! dn/dlogM = f*rho_0/M * dln(sigma^-1)/dlogM
   jakobian  = (log(f(i-1,2)) - log(f(i+1,2))) / (2.0d0*dM)
   mf(i-1,1) = f(i,2)
   mf(i-1,2) = f(i,3)
   mf(i-1,3) = f(i,1)
   mf(i-1,4) = f(i,3)*rho_0/f(i,1) * jakobian
enddo

end subroutine main
