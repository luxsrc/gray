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

pro plottv, img, sz, out, position=position, unit=unit, xTickName=xTickName

  p   = 2540 * position
  d   = 2.57036943d22
  rS  = 6.6725985d-8 * 4.3d6 * 1.99d33 / (2.99792458d10)^2
  mas = sz * (rS/d) * (180/!pi) * 60 * 60 * 1000
  out = mas

  if not keyword_set(xTickName) then xTickName = []

  plot, [0,1], [0,1], /nodata, /noerase, color=0,               $
        xRange=mas*[0.5,-0.5], yRange=mas*[-0.5,0.5],           $
        xTitle=textoidl('Relative right ascension ('+unit+')'), $
        yTitle=textoidl('Relative declination ('    +unit+')'), $
        xTickName=xTickName,                                    $
        /iso, /xStyle, /yStyle, position=p, /device

  tv, byte(256 * img < 255), p[0], p[1], xSize=p[2]-p[0]

  plot, [0,1], [0,1], /nodata, /noerase, color=255,                 $
        xRange=mas*[0.5,-0.5], yRange=mas*[-0.5,0.5],               $
        xTickname=replicate(' ', 16), yTickName=replicate(' ', 16), $
        /iso, /xStyle, /yStyle, position=p, /device

end
