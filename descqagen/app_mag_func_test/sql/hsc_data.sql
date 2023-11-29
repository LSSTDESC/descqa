SELECT
  meas.object_id
, meas.parent_id
, meas.ira
, meas.idec
, meas.icmodel_mag             as imag_cmodel
, meas.icmodel_mag_err         as imag_cmodel_err
, meas.icmodel_flux_flags      as iflux_cmodel_flags
, meas.icmodel_flux            as iflux_cmodel
, meas.icmodel_flux_err        as iflux_cmodel_err
, forced.merge_measurement_i
, forced.a_g
, forced.a_r
, forced.a_i
, forced.a_z
, forced.a_y
, forced.rcmodel_mag        as rmag_forced_cmodel
, forced.rcmodel_mag_err    as rmag_forced_cmodel_err
, forced.rcmodel_flux       as rflux_forced_cmodel
, forced.rcmodel_flux_err   as rflux_forced_cmodel_err
, forced.rcmodel_flux_flags as rflux_forced_cmodel_flags
, forced.icmodel_mag        as imag_forced_cmodel
, forced.icmodel_mag_err    as imag_forced_cmodel_err
, forced.icmodel_flux       as iflux_forced_cmodel
, forced.icmodel_flux_err   as iflux_forced_cmodel_err
, forced.icmodel_flux_flags as iflux_forced_cmodel_flags
, forced.gcmodel_mag        as gmag_forced_cmodel
, forced.zcmodel_mag        as zmag_forced_cmodel
, forced.ycmodel_mag        as ymag_forced_cmodel
, meas.tract
, meas.patch
, meas.gcountinputs
, meas.rcountinputs
, meas.icountinputs
, meas.zcountinputs
, meas.ycountinputs
, meas.iflags_pixel_bright_object_center
, meas.iflags_pixel_bright_object_any
, meas.iblendedness_flags
, meas.iblendedness_abs_flux

FROM
pdr1_deep.meas as meas
LEFT JOIN pdr1_deep.forced as forced using (object_id)
WHERE
-- Please uncomment to get a field you want

-- AEGIS
-- s16a_wide2.search_aegis(meas.patch_id)           AND

-- HECTOMAP
-- s16a_wide2.search_hectomap(meas.patch_id)        AND

-- GAMA09H
-- s16a_wide2.search_gama09h(meas.patch_id)         AND

-- WIDE12H
-- s16a_wide2.search_wide12h(meas.patch_id)         AND

-- GAMA15H
-- s16a_wide2.search_gama15h(meas.patch_id)         AND

-- VVDS
-- s16a_wide2.search_vvds(meas.patch_id)            AND

-- XMM
-- s16a_wide2.search_xmm(meas.patch_id)             AND

 NOT meas.ideblend_skipped                  AND
 NOT meas.iflags_badcentroid                AND
 NOT meas.iflags_pixel_edge                 AND
 NOT meas.iflags_pixel_interpolated_center  AND
 NOT meas.iflags_pixel_saturated_center     AND
 NOT meas.iflags_pixel_cr_center            AND
 NOT meas.iflags_pixel_bad                  AND
 NOT meas.iflags_pixel_suspect_center       AND
 NOT meas.iflags_pixel_clipped_any          AND
 meas.idetect_is_primary             	    AND
 meas.iclassification_extendedness != 0     
 ORDER BY meas.object_id
;