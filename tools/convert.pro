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

function convert, s, Jy=Jy

  a = s.size * 6.6725985d-8 * 4.3d6 * 1.99d33 / (2.99792458d10)^2
  I = total(total(s.img,1),1) / n_elements(s.img[*,*,0])
  d = 2.57036943d22

  if keyword_set(Jy) then $
    return, 1d23 * (a / d)^2 * I $
  else $
    return, 4d * !pi * (a)^2 * I * s.nu

end
