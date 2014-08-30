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

pro cleanup

  common static, org_device, org_font

  device, /close
  set_plot, org_device
  !p.font = org_font

end

pro setup, name, sz

  common static, org_device, org_font
  org_device = !d.name
  org_font   = !p.font

  set_plot, 'ps'
  device, filename=name+'.eps', /encap
  device, /inch, xSize=sz[0], ySize=sz[1]
  device, /color, bits_per_pixel=8

  !p.font = 0
  device, set_font='Times-Roman'
  device, /Times,          font_index=3
  device, /Times, /Italic, font_index=4

end
