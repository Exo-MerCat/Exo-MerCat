# def twomass_main_id(self) -> None:
#     """
#     The twomass_main_id function is used to find the 2MASS ID of a star.
#     For each target still missing the main_id, it takes the list of hosts and aliases.
#     If any of these has "2MASS" in the name, it searches the 2MASS catalog and
#     returns the 2MASS ID if it exists.
#     """
#
#     for i in self.data[self.data.main_id == ""].index:
#         list_of_missing = (
#             str(self.data.at[i, "host"]) + "," + str(self.data.at[i, "alias"])
#         )
#         for identifier in list_of_missing.split(","):
#             if "2MASS" in identifier:
#                 identifier = (
#                     identifier.replace("2MASS ", "")
#                     .replace("2MASSW ", " ")
#                     .replace("J", "")
#                     .replace('"', "")
#                 )
#
#                 service = pyvo.dal.TAPService(
#                     "http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap/"
#                 )
#                 query = (
#                     'SELECT  t.RAJ2000,  t.DEJ2000,  t."2MASS"'
#                     + ' FROM "II/246/out" as t WHERE t."2MASS" = "'
#                     + str(identifier)
#                     + '"'
#                 )
#
#                 response = service.run_sync(query, timeout=None)
#                 table = response.to_table().to_pandas()
#                 if len(table) > 0:
#                     if len(table) > 0:
#                         self.data.loc[i, "main_id"] = "2MASS " + str(
#                             table[table["2MASS"] == int(identifier)].ID.to_numpy()[
#                                 0
#                             ]
#                         )
#                         self.data.loc[i, "ra_simbad"] = table[
#                             table.ID == int(identifier)
#                         ].RAJ2000.to_numpy()[0]
#                         self.data.loc[i, "dec_simbad"] = table[
#                             table.ID == int(identifier)
#                         ].DEJ2000.to_numpy()[0]
#                         break
#
#     print("2MASS check unsuccessful", self.data[self.data.main_id == ""].shape)

# def gaia_main_id(self) -> None:
#     """
#     The gaia_main_id function takes the dataframe and searches for missing main_id values.
#     It then checks if there is a Gaia DR2 ID in the host or alias columns, and if so,
#     it uses that to query VizieR for the corresponding source_id. If found, it adds this
#     value to main_id column of the dataframe.
#     """
#
#     for i in self.data[self.data.main_id == ""].index:
#         list_of_missing = (
#             str(self.data.at[i, "host"]) + "," + str(self.data.at[i, "alias"])
#         )
#         for identifier in list_of_missing.split(","):
#             if "Gaia DR2" in identifier:
#                 identifier = identifier.replace("Gaia DR2 ", "")
#
#                 service = pyvo.dal.TAPService(
#                     "http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap/"
#                 )
#                 query = (
#                     "SELECT t.designation,  t.ra_epoch2000,  t.solution_id, "
#                     + ' t.dec_epoch2000 FROM "I/345/gaia2" as t WHERE t."source_id" = '
#                     + str(identifier)
#                 )
#
#                 response = service.run_sync(query, timeout=None)
#
#                 table = response.to_table().to_pandas()
#                 if len(table) > 0:
#                     self.data.loc[i, "main_id"] = str(
#                         table[
#                             table["designation"] == int(identifier)
#                         ].designation.to_numpy()[0]
#                     )
#                     self.data.loc[i, "ra_simbad"] = table[
#                         table["source_id"] == int(identifier)
#                     ].ra_epoch2000.to_numpy()[0]
#                     self.data.loc[i, "dec_simbad"] = table[
#                         table["source_id"] == int(identifier)
#                     ].dec_epoch2000.to_numpy()[0]
#                     break
#
#     logging.info(
#         "Gaia check unsuccessful " + str(self.data[self.data.main_id == ""].shape)
#     )
#
# def tess_main_id(self) -> None:
#     """
#     The tess_main_id function takes the dataframe and checks if there is a TIC ID
#     in the alias column. If so, it will query the TESS Input Catalog for that ID
#     and return all of its aliases. It then adds those aliases to the alias column
#     of self.data.
#     """
#
#     for i in self.data[self.data.main_id == ""].index:
#         list_of_missing = (
#             str(self.data.at[i, "host"]) + "," + str(self.data.at[i, "alias"])
#         )
#         for identifier in list_of_missing.split(","):
#             if "TIC" in identifier:
#                 identifier = identifier.replace("TIC ", "")
#
#                 service = pyvo.dal.TAPService(
#                     "http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap/"
#                 )
#                 query = (
#                     'SELECT t.TIC,  t.DEJ2000,  t.HIP,  t.UCAC4,  t."2MASS", '
#                     + " t.WISEA,  t.GAIA, t.KIC,  t.ID,  t.objID,  t.RAJ2000, "
#                     + ' t.TYC FROM "IV/38/tic" as t WHERE t."TIC" = '
#                     + str(identifier)
#                 )
#
#                 response = service.run_sync(query, timeout=None)
#
#                 table = response.to_table().to_pandas()
#                 if len(table) > 0:
#                     self.data.loc[i, "main_id"] = "TIC " + str(
#                         table[table["TIC"] == int(identifier)].TIC.to_numpy()[0]
#                     )
#                     self.data.loc[i, "ra_simbad"] = table[
#                         table["TIC"] == int(identifier)
#                     ].RAJ2000.to_numpy()[0]
#                     self.data.loc[i, "dec_simbad"] = table[
#                         table["TIC"] == int(identifier)
#                     ].DEJ2000.to_numpy()[0]
#                     if (
#                         str(
#                             table[table.TIC == int(identifier)]["2MASS"].to_numpy()[
#                                 0
#                             ]
#                         )
#                         != "<NA>"
#                     ):
#                         self.data.loc[i, "alias"] = (
#                             self.data.loc[i, "alias"]
#                             + ", 2MASS J"
#                             + str(
#                                 table[table.TIC == int(identifier)][
#                                     "2MASS"
#                                 ].to_numpy()[0]
#                             )
#                         )
#                     if (
#                         str(
#                             table[table.TIC == int(identifier)]["HIP"].to_numpy()[0]
#                         )
#                         != "<NA>"
#                     ):
#                         self.data.loc[i, "alias"] = (
#                             self.data.loc[i, "alias"]
#                             + ", HIP "
#                             + str(
#                                 table[table.TIC == int(identifier)].HIP.to_numpy()[
#                                     0
#                                 ]
#                             )
#                         )
#                     if (
#                         str(
#                             table[table.TIC == int(identifier)]["GAIA"].to_numpy()[
#                                 0
#                             ]
#                         )
#                         != "<NA>"
#                     ):
#                         self.data.loc[i, "alias"] = (
#                             self.data.loc[i, "alias"]
#                             + ", Gaia DR2 "
#                             + str(
#                                 table[table.TIC == int(identifier)].GAIA.to_numpy()[
#                                     0
#                                 ]
#                             )
#                         )
#                     if (
#                         str(
#                             table[table.TIC == int(identifier)]["KIC"].to_numpy()[0]
#                         )
#                         != "<NA>"
#                     ):
#                         self.data.loc[i, "alias"] = (
#                             self.data.loc[i, "alias"]
#                             + ", KIC "
#                             + str(
#                                 table[table.TIC == int(identifier)].KIC.to_numpy()[
#                                     0
#                                 ]
#                             )
#                         )
#
#                     break
#
#     logging.info(
#         "TESS check unsuccessful " + str(self.data[self.data.main_id == ""].shape)
#     )
#
# def epic_main_id(self) -> None:
#     """
# #     The epic_main_id function takes the dataframe and searches for EPIC IDs
#     in the host column. If it finds an EPIC ID, it will search for that ID
#     in VizieR and return a table of information about that star. It then adds
#     this information to the dataframe.
#     """
#
#     for i in self.data[self.data.main_id == ""].index:
#         list_of_missing = (
#             str(self.data.at[i, "host"]) + "," + str(self.data.at[i, "alias"])
#         )
#         for identifier in list_of_missing.split(","):
#             if "EPIC" in identifier:
#                 identifier = identifier.replace("EPIC ", "")
#
#                 service = pyvo.dal.TAPService(
#                     "http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap/"
#                 )
#                 query = (
#                     "SELECT  t.ID,  t.RAJ2000,  t.DEJ2000,  t.HIP,  t.TYC, "
#                     + 't.UCAC4,  t."2MASS", t.SDSS  FROM "IV/34/epic" as t WHERE t.ID = '
#                     + str(identifier)
#                 )
#
#                 try:
#                     response = service.run_sync(query, timeout=None)
#
#                     table = response.to_table().to_pandas()
#                     if len(table) > 0:
#                         self.data.loc[i, "main_id"] = "EPIC " + str(
#                             table[table.ID == int(identifier)].ID.to_numpy()[0]
#                         )
#                         self.data.loc[i, "ra_simbad"] = table[
#                             table.ID == int(identifier)
#                         ].RAJ2000.to_numpy()[0]
#                         self.data.loc[i, "dec_simbad"] = table[
#                             table.ID == int(identifier)
#                         ].DEJ2000.to_numpy()[0]
#                         if (
#                             str(
#                                 table[table.ID == int(identifier)][
#                                     "2MASS"
#                                 ].to_numpy()[0]
#                             )
#                             != "<NA>"
#                         ):
#                             self.data.loc[i, "alias"] = (
#                                 self.data.loc[i, "alias"]
#                                 + ", 2MASS "
#                                 + str(
#                                     table[table.ID == int(identifier)][
#                                         "2MASS"
#                                     ].to_numpy()[0]
#                                 )
#                             )
#                         if (
#                             str(
#                                 table[table.ID == int(identifier)][
#                                     "HIP"
#                                 ].to_numpy()[0]
#                             )
#                             != "<NA>"
#                         ):
#                             self.data.loc[i, "alias"] = (
#                                 self.data.loc[i, "alias"]
#                                 + ", HIP "
#                                 + str(
#                                     table[
#                                         table.ID == int(identifier)
#                                     ].HIP.to_numpy()[0]
#                                 )
#                             )
#                         break
#                 except:
#                     pass
#
#     logging.info(
#         "EPIC check unsuccessful " + str(self.data[self.data.main_id == ""].shape)
#     )
