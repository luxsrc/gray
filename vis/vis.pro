; Copyright (C) 2012 Chi-kwan Chan
; Copyright (C) 2012 Steward Observatory
;
; This file is part of geode.
;
; Geode is free software: you can redistribute it and/or modify it
; under the terms of the GNU General Public License as published by
; the Free Software Foundation, either version 3 of the License, or
; (at your option) any later version.
;
; Geode is distributed in the hope that it will be useful, but WITHOUT
; ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
; or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
; License for more details.
;
; You should have received a copy of the GNU General Public License
; along with geode.  If not, see <http://www.gnu.org/licenses/>.

function load, i

  f = string(i, format='(i04)') + '.raw'

  openr, lun, f, /get_lun
  time = 0.0d         & readu, lun, time
  m    = 0L           & readu, lun, m
  n    = 0L           & readu, lun, n
  data = fltarr(m, n) & readu, lun, data
  close, lun & free_lun, lun

  print, time

  x = reform(data[0,*])
  y = reform(data[1,*])
  z = reform(data[2,*])

  return, {t:time, x:x, y:y, z:z}

end

pro vis

  n = 16
  m = 100

  x = fltarr(m,n)
  y = fltarr(m,n)
  z = fltarr(m,n)

  for i = 0, m-1 do begin
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
