; Copyright (C) 2014 Chi-kwan Chan
; Copyright (C) 2014 Steward Observatory
;
; This file is part of GRay.
;
; GRay is free software: you can redistribute it and/or modify it
; under the terms of the GNU General Public License as published by
; the Free Software Foundation, either version 3 of the License, or
; (at your option) any later version.
;
; GRay is distributed in the hope that it will be useful, but WITHOUT
; ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
; or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
; License for more details.
;
; You should have received a copy of the GNU General Public License
; along with GRay.  If not, see <http://www.gnu.org/licenses/>.

function compose, l, s

  sz1 = size(l.img) & n_nu = sz1[0] eq 3 ? sz1[3] : 1 & sz1 = sz1[1:2]
  sz2 = size(s.img) & n_mu = sz2[0] eq 3 ? sz2[3] : 1 & sz2 = sz2[1:2]

  if n_nu ne n_mu or max(l.nu ne l.nu) or l.size le s.size then $
    return, {}

  sz2 = round(sz1 * s.size / l.size)
  b   = (sz1-sz2)/2
  e   =  sz1-b-1
  img = l.img
  img[b[0]:e[0],b[1]:e[1],*] = rebin(s.img, [sz2,n_nu])
  
  return, {size:l.size, nu:l.nu, img:img}

end
