// Compute Z masses 
ROOT::RVecF compute_z_masses_4l(const RVec<RVec<size_t>> &idx, cRVecF pt, cRVecF eta, cRVecF phi, cRVecF mass)
{
   ROOT::RVecF z_masses(2);
   for (size_t i = 0; i < 2; i++) {
      const auto i1 = idx[i][0]; const auto i2 = idx[i][1];
      ROOT::Math::PtEtaPhiMVector p1(pt[i1], eta[i1], phi[i1], mass[i1]);
      ROOT::Math::PtEtaPhiMVector p2(pt[i2], eta[i2], phi[i2], mass[i2]);
      z_masses[i] = (p1 + p2).M();
   }
   return z_masses
}











