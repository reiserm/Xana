subroutine hist_adu_rfs( mt, mk, thr, ph, ppd, nx, ny, h1)

    implicit none
!   Input arguments:
    integer(kind=4), intent(in) :: thr, ph, nx, ny, ppd    
    real(kind=8), intent(in), dimension(nx,ny) :: mt
    integer(kind=4), intent(in), dimension(nx,ny) :: mk

!   Output arguments
    integer(kind=4), dimension(nx,ny) :: dar
    real(kind=8), intent(out), dimension(nx*ny) :: h1
        
!   Variables used only in subroutine
    integer(kind=4), dimension(nx,ny) :: mto
    integer :: ifs, ix, iy, st
    integer :: cxj, cyj, iw, j, jjj, ct1
    integer(kind=4), dimension(ppd) :: cx, cy
    real(kind=8) :: mpx, rfs

!   this is to dropletize the hole image!!!      
!   for wxpcs
!   msumpix, mpix, drop_img = dropimgood_nn(mt,dkimg,lth,bADU,tADU,mNp,ph,nx,ny)
    print *, nx, ny
    ct1 = 1
    j = 1
    mpx = 0.0 
    st = -1 
    iw = 0
!   Subtract dark (dk) from matrix (mt)
!   and mark pixels below threshold (thr) in mto
    do iy=1,ny,1
        do ix=1,nx,1
            dar(ix,iy)=0 
            mto(ix,iy)=0
            if (mt(ix,iy) .LT. thr .OR. mk(ix,iy) .EQ. 1) then 
                mto(ix,iy) = 1
            endif
        end do
    end do
!   Scan matrix for droplets and
!   save the indices of a droplet in cx and cy.      
    do iy=1,ny,1
        do ix=1,nx,1
            if (mto(ix,iy) .EQ. 0) then
                mto(ix,iy)=1
                st=0
                cx(j)=ix
                cy(j)=iy
                iw=0
            endif  
            do while (st .GE. 0)
                iw=iw+1
                cxj=cx(iw)
                cyj=cy(iw)  
                st=st-1
                if (cxj-1 .GE. 1) then
                    if (mto(cxj-1,cyj) .EQ. 0) then      
                        mto(cxj-1,cyj)=1
                        j=j+1
                        cx(j)=cxj-1
                        cy(j)=cyj
                        st=st+1 
                    endif
                endif
                if (cxj+1 .LE. nx) then
                    if (mto(cxj+1,cyj) .EQ. 0) then      
                        mto(cxj+1,cyj)=1
                        j=j+1
                        cx(j)=cxj+1
                        cy(j)=cyj 
                        st=st+1
                    endif
                endif
                if (cyj-1 .GE. 1) then 
                    if (mto(cxj,cyj-1) .EQ. 0) then      
                        mto(cxj,cyj-1)=1
                        j=j+1
                        cx(j)=cxj
                        cy(j)=cyj-1  
                        st=st+1  
                    endif
                endif
                if (cyj+1 .LE. ny) then
                    if (mto(cxj,cyj+1) .EQ. 0) then      
                        mto(cxj,cyj+1)=1
                        j=j+1
                        cx(j)=cxj
                        cy(j)=cyj+1 
                        st=st+1 
                    endif
                endif
            enddo
!           Gather the intensity of a droplet in rfs.
!           Matrix dar contains photon positions.
            if (iw .GE. 1 .AND. iw .LE. ppd) then
                rfs=0
                do jjj=1,iw,1
                    rfs=rfs+mt(cx(jjj),cy(jjj))
                end do
                h1(ct1) = rfs
                ct1 = ct1 + 1
                print *, ct1
                ifs=nint(rfs/ph)
            endif 
            do jjj=1,iw,1
!               mt(cx(jjj),cy(jjj))=0.0
               mto(cx(jjj),cy(jjj))=1
               cx(jjj)=0
               cy(jjj)=0
            end do    
            j=1
            st=-1
            iw=0 
        end do
    end do

end subroutine hist_adu_rfs
