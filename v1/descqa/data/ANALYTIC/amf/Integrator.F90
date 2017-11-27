!--------------------------------------------------------
! Integration using Simpson's rule

! use as: result = integrate(func, a, b, tol)

! where:
!         func is the integrating function
!         a, b are limits for integration
!         tol is the (optional, default 1.0E-4) tolerance


module ModuleIntegrator

implicit none
public  :: integrate

contains

double precision function integrate(f, a, b, tol)
implicit none
double precision, intent(in) :: a, b
double precision, optional, intent(in) :: tol
interface
   double precision function f(x)
   double precision, intent(in) :: x
   end function f
end interface
double precision :: old, eps, h, x
integer, parameter :: JMAX=20
integer :: i, j, n

if (present(tol)) then
   eps = tol
else
   eps = 1.0d-4
endif

old = -1.0d0
n   = 1
do j = 1, JMAX
   n = n*2
   h = (b-a) / n
   integrate = 0.0d0
   do i = 2, n-2, 2
      x = a + i*h
      integrate = integrate + 2.0d0*f(x) + 4.0d0*f(x+h)
   enddo
   integrate = (integrate + f(a) + f(b) + 4.0d0*f(a+h)) * h / 3.0d0

   if (j > 4) then  ! prevent accidental convergence
      if (abs(integrate-old) .le. eps*abs(old)) RETURN
   endif
   old = integrate
enddo
STOP "Integrator: no convergence!"
end function integrate

end module ModuleIntegrator
