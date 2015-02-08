; Copyright (C) 2015 Chi-kwan Chan
; Copyright (C) 2015 Steward Observatory
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

pro merge_raw, in, out

  data = [load_raw(in[0])]
  for i = 1, n_elements(in)-1 do $
    data = [data, load_raw(in[i])]

  flux = data[0].flux
  for i = 1, n_elements(data)-1 do $
    flux = [flux, data[i].flux]

  nu = data[0].nu
  for i = 1, n_elements(data)-1 do $
    nu = [nu, data[i].nu]

  openw, lun, out, /get_lun

    ascii = [flux, nu, data[0].size]
    printf, lun, ascii, format='('+string(n_elements(ascii))+'(e15.9,:,%"\t"))'

    writeu, lun, long64(n_elements(data[0].img[*,*,0]))

    foreach d, data do $
       writeu, lun, float(d.img)

  close, lun & free_lun, lun

end
