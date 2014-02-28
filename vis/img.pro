; Copyright (C) 2012,2013 Chi-kwan Chan
; Copyright (C) 2012,2013 Steward Observatory
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

function load, i

  f = string(i, format='(i04)') + '.raw'

  openr, lun, f, /get_lun
  t = 0.0d         & readu, lun, t
  m = 0LL          & readu, lun, m
  n = 0LL          & readu, lun, n
  d = fltarr(m, n) & readu, lun, d
  close, lun & free_lun, lun

  t     = reform(d[ 0,*])
  r     = reform(d[ 1,*])
  theta = reform(d[ 2,*])
  phi   = reform(d[ 3,*])
  Fr    = reform(d[ 9,*])
  Fg    = reform(d[10,*])
  Fb    = reform(d[11,*])
  
  r_cyl = r * sin(theta)
  x     = r_cyl * cos(phi)
  y     = r_cyl * sin(phi)
  z     = r * cos(theta)
  
  return, {t:t, x:x, y:y, z:z, R:Fr, G:Fg, B:Fb}

end

pro img

  d = load(0)
  R = reform(d.R, 512, 512)
  G = reform(d.G, 512, 512)
  B = reform(d.B, 512, 512)

  print, min(R), max(R)
  print, min(G), max(G)
  print, min(B), max(B)

  !p.multi=[0,2,2]
  shade_surf, R, ax=80
  shade_surf, G, ax=80
  shade_surf, B, ax=80

end
