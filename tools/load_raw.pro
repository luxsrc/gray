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

function load_raw, name

  if strmid(name,strlen(name)-4,4) ne '.raw' then name = name+'.raw'
  spawn, 'head -qn1 '+name+' | wc -w', w
  n_nu = fix(w)-1

  openr, lun, name, /get_lun
  h = dblarr(n_nu+1) & readf, lun, h
  n = 0LL            & readu, lun, n
  I = fltarr(n,n_nu) & readu, lun, I
  close, lun & free_lun, lun

  n1 = long64(sqrt(n))
  n2 = n / n1
  I  = reform(double(I),n1,n2,n_nu)

  return, {size:h[0], nu:h[1:*], flux:I}

end
