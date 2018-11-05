subroutine hist_adu_fss( mt, dk, mk, thr, lt, ut, ph, nx, ny, ppd, h1 )

    implicit none
!   Input arguments:
    integer(kind=4), intent(in) :: thr, lt, ut, ph, nx, ny, ppd    
    real(kind=8), intent(inout), dimension(nx,ny) :: mt
    real(kind=8), intent(in), dimension(nx,ny) :: dk
    integer(kind=4), intent(in), dimension(nx,ny) :: mk

!   Output arguments
    integer(kind=4), dimension(nx,ny) :: dar
    real(kind=8), intent(out), dimension(nx*ny) :: h1
        
!   Variables used only in subroutine
    integer(kind=4), dimension(nx,ny) :: mto
    integer :: ifs, ix, iy, st, mx, my, ixx, iyy, bx, ex, by, ey
    integer :: cxj, cyj, iw, j, jjj, cmkp, ct1
    integer(kind=4), dimension(nx*ny) :: cx, cy
    real(kind=8) :: fss, mpx, rfs
    
    !f2py intent(in,out) :: mt

!   this is to dropletize the hole image!!!      
!   for wxpcs
!   msumpix, mpix, drop_img = dropimgood_nn(mt,dkimg,lth,bADU,tADU,mNp,ph,nx,ny)
    ct1 = 1
    j = 1
    fss = 0.0
    mpx = 0.0 
    mx = 0
    my = 0 
    st = -1 
    iw = 0
    cmkp = 1
!   Subtract dark (dk) from matrix (mt)
!   and mark pixels below threshold (thr) in mto
    do iy=1,ny,1
        do ix=1,nx,1
            mt(ix,iy)=mt(ix,iy)-dk(ix,iy)
            dar(ix,iy)=0 
            mto(ix,iy)=0
            if (mt(ix,iy) .LT. thr .OR. mk(ix,iy) .EQ. 1) then 
                mt(ix,iy) = 0.0
                mto(ix,iy) = 1
            endif
            if (mk(ix,iy) .EQ. 1) then
                cmkp = cmkp + 1
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
                ifs=nint(rfs/ph)
                do while (ifs .GT. 0)
                    mpx=0.0
                    do jjj=1,iw,1
                        if (mt(cx(jjj),cy(jjj)) .GT. mpx) then
                            if (mto(cx(jjj),cy(jjj)) .EQ. 1) then
                                mpx=mt(cx(jjj),cy(jjj))
                                mx=cx(jjj)
                                my=cy(jjj)
                            endif
                        endif
                    end do
                    if (mx .EQ. 1) then
                        bx=mx
                        ex=mx+1      
                    elseif (mx .EQ. nx) then
                        bx=mx-1
                        ex=mx
                    else
                        bx=mx-1
                        ex=mx+1
                    endif
                    if (my .EQ. 1) then
                        by=my
                        ey=my+1
                    elseif (my .EQ. ny) then
                        by=my-1
                        ey=my
                    else
                        by=my-1
                        ey=my+1
                    endif
                    fss=0.0
                    do iyy=by,ey,1
                        fss=fss+mt(mx,iyy)
                    end do
                    do ixx=bx,ex,1
                        fss=fss+mt(ixx,my)
                    end do
                    h1(ct1) = fss
                    ct1 = ct1 + 1
                    fss = fss-mt(mx,my)
                    if (fss .GE. lt .AND. fss .LE. ut) then 
                        dar(mx,my)=1
                        mt(mx,my)=0.0
                        mto(mx,my)=1
                        ifs=ifs-1
                        rfs=rfs-ph
                    elseif (fss .GT. ut) then
                        if (mpx .LE. ut) then 
                            dar(mx,my)=1
                            mt(mx,my)=0.0
                            mto(mx,my)=1
                        elseif (mpx .GT. ut) then 
                            dar(mx,my)=int(fss/ph)
                            mt(mx,my)=mt(mx,my)-dar(mx,my)*ph
                                if (mt(mx,my) .LT. 0) then
                                    mt(mx,my)=0.0
                                    mto(mx,my)=1
                                else
                                    mto(mx,my)=0
                                endif
                        endif 
                        ifs=ifs-dar(mx,my)
                        rfs=rfs-ph*dar(mx,my) 
                    else
                        if (rfs .LT. lt .OR. mpx .EQ. 0.0) then 
                            mt(mx,my)=0.0
                            mto(mx,my)=1
                            ifs=0
                        else
                            mt(mx,my)=0.0
                            mto(mx,my)=1
                        endif
                    endif 
                enddo
                do jjj=1,iw,1
                    mt(cx(jjj),cy(jjj))=0.0
                    mto(cx(jjj),cy(jjj))=1
                    cx(jjj)=0
                    cy(jjj)=0  
                end do
            elseif (iw .GT. ppd) then
                do jjj=1,iw,1
                    cmkp = cmkp + 1
                    mt(cx(jjj),cy(jjj))=0.0
                    mto(cx(jjj),cy(jjj))=1
                    cx(jjj)=0
                    cy(jjj)=0
                end do    
            endif
            j=1
            st=-1
            iw=0 
        end do
    end do
end subroutine hist_adu_fss
