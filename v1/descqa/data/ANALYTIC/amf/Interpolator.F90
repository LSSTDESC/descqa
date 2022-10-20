!--------------------------------------------------------------------------
! Linear interpolation

! use as:  y = interpolate(n, xx, yy, x)

! where: 
!        n is the size of arrays xx and yy
!        xx is the array of points (monotonically increasing or decreasing)
!        yy is the array of corresponding function values.
!        x is the point for which interpolation is needed


module ModuleInterpolator

implicit none
public  :: interpolate
private :: hunt

contains

double precision function interpolate(n, xx, yy, x)
implicit none
integer, intent(in) :: n
double precision, intent(in) :: x
double precision, dimension(n), intent(in) :: xx, yy
integer, save :: jlo

call hunt(xx, x, jlo)
!jlo = floor((x-xx(1)) / (xx(2)-xx(1)))
!if (x .ge. xx(jlo+1)) jlo = jlo+1

if ((x .gt. xx(n)) .or. (x .lt. xx(1))) &
     write(*,*) 'Warning! Extrapolating outside the tabulated range, x =', x

interpolate = (yy(jlo)-yy(jlo+1))/(xx(jlo)-xx(jlo+1)) * (x-xx(jlo)) + yy(jlo)

end function interpolate


! Table lookup from NR
subroutine hunt(xx, x, jlo)
implicit none
integer, intent(inout) :: jlo
double precision, intent(in) :: x
double precision, dimension(:), intent(in) :: xx
integer :: n, inc, jhi, jm
logical :: ascnd

n = size(xx)
ascnd = (xx(n) >= xx(1))

if (jlo <= 0 .or. jlo >= n) then  ! bisect
   jlo = 0
   jhi = n+1
else
   inc = 1
   if (x >= xx(jlo) .eqv. ascnd) then  ! hunt up
      do
         jhi = jlo+inc
         if (jhi > n) then  ! end of the table
            jhi = n+1
            EXIT
         else
            if (x < xx(jhi) .eqv. ascnd) EXIT
            jlo = jhi
            inc = inc+inc  ! double the increment
         endif
      enddo
   else  ! hunt down
      jhi = jlo
      do
         jlo = jhi - inc
         if (jlo < 1) then  ! end of the table
            jlo = 0
            EXIT
         else
            if (x >= xx(jlo) .eqv. ascnd) EXIT
            jhi = jlo
            inc = inc+inc
         endif
      enddo
   endif
endif

do
   if (jhi-jlo <= 1) then
      if (x == xx(n)) jlo=n-1
      if (x == xx(1)) jlo=1
      exit
   else
      jm = (jhi+jlo)/2
      if (x >= xx(jm) .eqv. ascnd) then
         jlo = jm
      else
         jhi = jm
      endif
   endif
enddo
end subroutine hunt

end module ModuleInterpolator
