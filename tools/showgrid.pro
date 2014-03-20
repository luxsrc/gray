pro showgrid, dir

  if n_elements(dir) eq 0 then dir = '.'

  openr, lun, dir + '/usgdump2d', /get_lun
  h        = bytarr(4)   & readu, lun, h
  t        = 0.0d        & readu, lun, t
  n        = lon64arr(3) & readu, lun, n
  sx       = dblarr(3)   & readu, lun, sx
  dx       = dblarr(3)   & readu, lun, dx
  nstep    = 1LL         & readu, lun, nstep
  gamma    = 0.0d        & readu, lun, gamma
  aspin    = 0.0d        & readu, lun, aspin
  R0       = 0.0d        & readu, lun, R0
  Rin      = 0.0d        & readu, lun, Rin
  Rout     = 0.0d        & readu, lun, Rout
  hslope   = 0.0d        & readu, lun, hslope
  dt       = 0.0d        & readu, lun, dt
  defcoord = bytarr(8)   & readu, lun, defcoord
  h        = bytarr(4)   & readu, lun, h
  cells    = replicate({cell,                                            $
                        h1:bytarr(4),                                    $
                        i:lon64arr(3),    x:dblarr(3),      v:dblarr(3), $
                        gcon:dblarr(4,4), gcov:dblarr(4,4), gdet:0.0d,   $
                        ck:dblarr(4),     dxdxp:dblarr(4,4),             $
                        tmp:dblarr(6),    h2:bytarr(4)}, n[0], n[1])
  readu, lun, cells
  close, lun & free_lun, lun

  print, n
  print, sx, dx
  print, R0, Rin, Rout
  print, gamma, aspin, hslope

  !p.multi=[0,2,2]

  print, 'min:', min(cells.v[0]), ', max:', max(cells.v[0])
  surface, alog10(cells.v[0])

  ;print, where(cells[*,0].v[0] gt 1e3)
  plot, cells[*,0].v[0], /yLog

  print, 'min:', min(cells.v[1]), ', max:', max(cells.v[1])
  surface, cells.v[1]

  plot, cells[0,*].v[1], yRange=[0,3.14]
  for i = 10, 263, 10 do oplot, cells[i,*].v[1]

end
