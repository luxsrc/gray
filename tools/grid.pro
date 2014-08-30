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

function unit, w, n

  d  = 2.57036943e22
  rS = 6.6725985e-8 * 4.3e6 * 1.99e33 / (2.99792458e10)^2

  return, (double(w)/n) * (rS/d)

end

function grid, sz, n1, n2

  if n_elements(n1) eq 0 then n1 = 512
  if n_elements(n2) eq 0 then n2 = n1

  a_mas = unit(sz,n1) * (180/!pi) * 60 * 60 * 1000
  b_mas = unit(sz,n2) * (180/!pi) * 60 * 60 * 1000

  return, {a:-(dindgen(n1) - 0.5*(n1-1)) # (dblarr(n2) + a_mas), $
           b: (dblarr(n1) + b_mas) # (dindgen(n2) - 0.5*(n2-1))}
end
