# docs — guides and papers

[`getting_the_data.md`](getting_the_data.md) explains how to download the
public archive the notebooks and scripts use: one command, by hand, checksums,
folder layout and troubleshooting.

`papers/` holds the published papers each module implements, so the code can
be checked against its source. Only the open-access ones (CC BY) are here.

| file | paper | implemented in |
|---|---|---|
| [`papers/2025_Mehri_IJGI_CADP.pdf`](papers/2025_Mehri_IJGI_CADP.pdf) | Mehri, Hooshangi & Mahdizadeh Gharakhanlou (2025). *A Novel Context-Aware Douglas–Peucker (CADP) Trajectory Compression Method.* ISPRS Int. J. Geo-Inf. 14(2), 58. [doi:10.3390/ijgi14020058](https://doi.org/10.3390/ijgi14020058) | `catraj.compression` |
| [`papers/2021_Mehri_IEEE-Access_Contextual-Hybrid-Model_CPLS-CLSTM.pdf`](papers/2021_Mehri_IEEE-Access_Contextual-Hybrid-Model_CPLS-CLSTM.pdf) | Mehri, Alesheikh & Basiri (2021). *A Contextual Hybrid Model for Vessel Movement Prediction.* IEEE Access 9, 45600–45613. [doi:10.1109/ACCESS.2021.3066463](https://doi.org/10.1109/ACCESS.2021.3066463) | CPLS, two-stage CPLS: `catraj.compression`; CLSTM (planned) |

The annotation method follows Mehri, Alesheikh & Basiri (2023). *A
context-aware approach for vessels' trajectory prediction.* Ocean Engineering
282, 114916, [doi:10.1016/j.oceaneng.2023.114916](https://doi.org/10.1016/j.oceaneng.2023.114916).
It is a subscription article, so it is linked rather than included.
