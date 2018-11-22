subroutine fecorrt3m(pix, t, cc, lpi, nt)
     implicit none
     !initialize variables
     integer(kind=4) :: i, t0, j
     integer(kind=4), intent(in) :: lpi, nt
     integer(kind=4), intent(in), dimension(0:lpi-1) :: pix, t
     integer(kind=4), intent(inout), dimension(0:nt-1,0:nt-1) :: cc
!f2py intent(in,out) cc      

!    fecorrt function to caclulate two-time cf from events for
!    eventcorrelator
     i=0
     do while (i .LT. lpi) 
       t0=t(i)
       j=i+1
       do while (pix(j) .EQ. pix(i)) 
         cc(t(j),t0)=cc(t(j),t0)+1
         j=j+1
         if (j .GE. lpi) exit
       end do
       i=i+1
     end do
     do i=0, nt-1
       do j=i, nt-1
         cc(j,i) = (cc(j,i) + cc(i,j))
         cc(i,j) = cc(j,i)  
       end do
     end do
     return
end subroutine fecorrt3m
