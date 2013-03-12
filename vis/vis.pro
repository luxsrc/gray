; Copyright (C) 2012 Chi-kwan Chan
; Copyright (C) 2012 Steward Observatory
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
  m = 0L           & readu, lun, m
  n = 0L           & readu, lun, n
  d = fltarr(m, n) & readu, lun, d
  close, lun & free_lun, lun

  if m eq 6 then begin
    x = reform(d[0,*])
    y = reform(d[1,*])
    z = reform(d[2,*])

    print, t
  end else begin
    t     = reform(d[0,*])
    r     = reform(d[1,*])
    theta = reform(d[2,*])
    phi   = reform(d[3,*])

    r_cyl = r * sin(theta)
    x     = r_cyl * cos(phi)
    y     = r_cyl * sin(phi)
    z     = r * cos(theta)

    print, min(t), max(t)
  endelse

  return, {t:t, x:x, y:y, z:z}

end

pro vis, t, n

  if n_elements(t) eq 0 then t = 16
  if n_elements(n) eq 0 then n = 16

  x = fltarr(t,n)
  y = fltarr(t,n)
  z = fltarr(t,n)

  for i = 0, t-1 do begin
    l      = load(i)
    x[i,*] = l.x[0:n-1]
    y[i,*] = l.y[0:n-1]
    z[i,*] = l.z[0:n-1]
  endfor

  r = [-50,50]
  surface, [[r],[r]], r, r, /nodata, /save
  for i = 0, n-1 do $
    plots, x[*,i], y[*,i], z[*,i], /t3d

end
