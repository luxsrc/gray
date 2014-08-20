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

function gaussian2, a, b, pa, nu, major, minor, upper=upper, lower=lower

  if n_elements(pa) eq 0 then pa = 78.0d
  if n_elements(nu) eq 0 then nu = 2.3060958d11

  r = (2.99792458d10 / nu)^2 / 2.35482004503d

  if n_elements(minor) eq 0 then minor = 0.640d * r
  if n_elements(major) eq 0 then major = 1.309d * r

  x =  a * sin(pa * !pi/180) + b * cos(pa * !pi/180)
  y = -a * cos(pa * !pi/180) + b * sin(pa * !pi/180)

  if keyword_set(upper) then $
    f = exp(-0.5d * ((x/major)^2 + (y/major)^2)) $
  else if keyword_set(lower) then $
    f = exp(-0.5d * ((x/minor)^2 + (y/minor)^2)) $
  else $
    f = exp(-0.5d * ((x/major)^2 + (y/minor)^2))

  return, f / total(f)

end

function scatter, data, pa, nu, noscatter=noscatter, upper=upper, lower=lower

  img  = double(data.img)
  n    = size(img)
  grid = grid(data.size, n[1], n[2])

  if keyword_set(noscatter) then begin
    return, {n:n[1:2], a:grid.a, b:grid.b, img:img}
  endif else begin
    kern = gaussian2(grid.a, grid.b, pa, nu, upper=upper, lower=lower)
    return, {n:n[1:2], a:grid.a, b:grid.b, img:float(convol_fft(img, kern))}
  endelse

end
