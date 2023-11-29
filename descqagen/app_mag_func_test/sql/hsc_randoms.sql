SELECT
  random.object_id
, random.ra
, random.dec
, random.tract
, random.patch
, random.gcountinputs
, random.rcountinputs
, random.icountinputs
, random.zcountinputs
, random.ycountinputs
, random.iflags_pixel_bright_object_center
, random.iflags_pixel_bright_object_any

FROM
pdr1_deep.random as random
WHERE
 NOT random.iflags_pixel_edge                 AND
 NOT random.iflags_pixel_interpolated_center  AND
 NOT random.iflags_pixel_saturated_center     AND
 NOT random.iflags_pixel_cr_center            AND
 NOT random.iflags_pixel_bad                  AND
 NOT random.iflags_pixel_suspect_center       AND
 random.idetect_is_primary             	    
 ORDER BY random.object_id
;