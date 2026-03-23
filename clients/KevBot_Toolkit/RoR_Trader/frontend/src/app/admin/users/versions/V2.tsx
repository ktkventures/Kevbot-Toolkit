'use client';

import { useState, useMemo } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import Modal from '@/components/Modal';
import MetricCard from '@/components/MetricCard';

interface User {
  id: string;
  name: string;
  email: string;
  role: string;
  joinDate: string;
  status: 'active' | 'inactive' | 'suspended';
  subscriptions: number;
  earnings: number;
  strategiesCreated: number;
  lastActive: string;
}

const mockUsers: User[] = [
  { id: 'u1', name: 'Alex Kim', email: 'alex.k@email.com', role: 'Portfolio Builder', joinDate: '2025-11-15', status: 'active', subscriptions: 128, earnings: 3840, strategiesCreated: 12, lastActive: '5m ago' },
  { id: 'u2', name: 'Sarah Mitchell', email: 'sarah.m@email.com', role: 'Pack Creator', joinDate: '2025-12-02', status: 'active', subscriptions: 0, earnings: 1800, strategiesCreated: 0, lastActive: '1h ago' },
  { id: 'u3', name: 'James Torres', email: 'james.t@email.com', role: 'Subscriber', joinDate: '2026-01-10', status: 'active', subscriptions: 3, earnings: 0, strategiesCreated: 0, lastActive: '2h ago' },
  { id: 'u4', name: 'Maria Lopez', email: 'maria.l@email.com', role: 'Strategy Builder', joinDate: '2026-01-22', status: 'inactive', subscriptions: 0, earnings: 960, strategiesCreated: 8, lastActive: '3d ago' },
  { id: 'u5', name: 'David Reed', email: 'david.r@email.com', role: 'Portfolio Builder', joinDate: '2026-02-05', status: 'active', subscriptions: 67, earnings: 2010, strategiesCreated: 5, lastActive: '20m ago' },
  { id: 'u6', name: 'Emily Chen', email: 'emily.c@email.com', role: 'Subscriber', joinDate: '2026-02-18', status: 'suspended', subscriptions: 1, earnings: 0, strategiesCreated: 0, lastActive: '7d ago' },
  { id: 'u7', name: 'Michael Brown', email: 'michael.b@email.com', role: 'Subscriber', joinDate: '2026-03-01', status: 'active', subscriptions: 2, earnings: 0, strategiesCreated: 0, lastActive: '15m ago' },
  { id: 'u8', name: 'Lisa Park', email: 'lisa.p@email.com', role: 'Strategy Builder', joinDate: '2026-03-08', status: 'active', subscriptions: 0, earnings: 480, strategiesCreated: 4, lastActive: '45m ago' },
];

const statusColors: Record<string, { color: string; bg: string }> = {
  active: { color: 'var(--green)', bg: 'var(--green-muted)' },
  inactive: { color: 'var(--text-muted)', bg: 'var(--bg-input)' },
  suspended: { color: 'var(--red)', bg: 'var(--red-muted)' },
};

const roles = ['All', 'Subscriber', 'Strategy Builder', 'Portfolio Builder', 'Pack Creator'];
const statuses = ['All', 'active', 'inactive', 'suspended'];

export default function AdminUsersV2() {
  const [search, setSearch] = useState('');
  const [roleFilter, setRoleFilter] = useState('All');
  const [statusFilter, setStatusFilter] = useState('All');
  const [selectedUser, setSelectedUser] = useState<User | null>(null);

  const filtered = useMemo(() => {
    return mockUsers.filter((u) => {
      if (search && !u.name.toLowerCase().includes(search.toLowerCase()) && !u.email.toLowerCase().includes(search.toLowerCase())) return false;
      if (roleFilter !== 'All' && u.role !== roleFilter) return false;
      if (statusFilter !== 'All' && u.status !== statusFilter) return false;
      return true;
    });
  }, [search, roleFilter, statusFilter]);

  return (
    <div>
      <PageHeader
        title="User Management"
        subtitle={`${mockUsers.length} total users`}
        backHref="/admin"
      />

      {/* Summary Metrics */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <MetricCard label="Total Users" value={mockUsers.length.toString()} />
        <MetricCard label="Active" value={mockUsers.filter((u) => u.status === 'active').length.toString()} />
        <MetricCard label="Creators" value={mockUsers.filter((u) => u.role !== 'Subscriber').length.toString()} />
        <MetricCard label="Suspended" value={mockUsers.filter((u) => u.status === 'suspended').length.toString()} />
      </div>

      {/* Filters */}
      <Card className="mb-5">
        <div className="flex flex-wrap gap-4">
          <div className="flex-1 min-w-[200px]">
            <input
              type="text"
              placeholder="Search by name or email..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full px-3 py-2 rounded-lg text-sm"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
            />
          </div>
          <select
            value={roleFilter}
            onChange={(e) => setRoleFilter(e.target.value)}
            className="px-3 py-2 rounded-lg text-sm"
            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
          >
            {roles.map((r) => <option key={r} value={r}>{r === 'All' ? 'All Roles' : r}</option>)}
          </select>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="px-3 py-2 rounded-lg text-sm"
            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
          >
            {statuses.map((s) => <option key={s} value={s}>{s === 'All' ? 'All Statuses' : s}</option>)}
          </select>
        </div>
      </Card>

      {/* User Table */}
      <Card>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border)' }}>
                <th className="text-left py-3 px-3 font-medium" style={{ color: 'var(--text-muted)' }}>User</th>
                <th className="text-left py-3 px-3 font-medium" style={{ color: 'var(--text-muted)' }}>Role</th>
                <th className="text-left py-3 px-3 font-medium" style={{ color: 'var(--text-muted)' }}>Joined</th>
                <th className="text-left py-3 px-3 font-medium" style={{ color: 'var(--text-muted)' }}>Last Active</th>
                <th className="text-left py-3 px-3 font-medium" style={{ color: 'var(--text-muted)' }}>Status</th>
                <th className="text-right py-3 px-3 font-medium" style={{ color: 'var(--text-muted)' }}>Earnings</th>
                <th className="text-center py-3 px-3 font-medium" style={{ color: 'var(--text-muted)' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((user) => (
                <tr
                  key={user.id}
                  className="cursor-pointer transition-colors"
                  style={{ borderBottom: '1px solid var(--border)' }}
                  onClick={() => setSelectedUser(user)}
                >
                  <td className="py-3 px-3">
                    <div className="flex items-center gap-3">
                      <div
                        className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-semibold shrink-0"
                        style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
                      >
                        {user.name.charAt(0)}
                      </div>
                      <div>
                        <p className="font-medium">{user.name}</p>
                        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{user.email}</p>
                      </div>
                    </div>
                  </td>
                  <td className="py-3 px-3" style={{ color: 'var(--text-secondary)' }}>{user.role}</td>
                  <td className="py-3 px-3" style={{ color: 'var(--text-muted)' }}>{user.joinDate}</td>
                  <td className="py-3 px-3" style={{ color: 'var(--text-muted)' }}>{user.lastActive}</td>
                  <td className="py-3 px-3">
                    <span
                      className="px-2 py-0.5 rounded-full text-xs font-medium"
                      style={{ background: statusColors[user.status].bg, color: statusColors[user.status].color }}
                    >
                      {user.status}
                    </span>
                  </td>
                  <td className="py-3 px-3 text-right font-medium">
                    {user.earnings > 0 ? `$${user.earnings.toLocaleString()}` : '--'}
                  </td>
                  <td className="py-3 px-3 text-center">
                    <button
                      className="text-xs px-2 py-1 rounded"
                      style={{ background: 'var(--bg-input)', color: 'var(--text-secondary)', border: '1px solid var(--border)' }}
                      onClick={(e) => { e.stopPropagation(); setSelectedUser(user); }}
                    >
                      View
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {filtered.length === 0 && (
            <p className="text-center py-8 text-sm" style={{ color: 'var(--text-muted)' }}>No users match your filters</p>
          )}
        </div>
      </Card>

      {/* User Detail Modal */}
      <Modal title={selectedUser?.name ?? 'User Detail'} isOpen={!!selectedUser} onClose={() => setSelectedUser(null)} width="560px">
        {selectedUser && (
          <div className="space-y-5">
            <div className="flex items-center gap-4">
              <div
                className="w-14 h-14 rounded-full flex items-center justify-center text-xl font-semibold"
                style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
              >
                {selectedUser.name.charAt(0)}
              </div>
              <div>
                <p className="font-semibold text-lg">{selectedUser.name}</p>
                <p className="text-sm" style={{ color: 'var(--text-muted)' }}>{selectedUser.email}</p>
                <div className="flex gap-2 mt-1">
                  <span className="px-2 py-0.5 rounded-full text-xs" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>{selectedUser.role}</span>
                  <span className="px-2 py-0.5 rounded-full text-xs" style={{ background: statusColors[selectedUser.status].bg, color: statusColors[selectedUser.status].color }}>{selectedUser.status}</span>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Subscriptions</p>
                <p className="text-lg font-semibold">{selectedUser.subscriptions}</p>
              </div>
              <div className="p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Earnings</p>
                <p className="text-lg font-semibold">${selectedUser.earnings.toLocaleString()}</p>
              </div>
              <div className="p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Strategies Created</p>
                <p className="text-lg font-semibold">{selectedUser.strategiesCreated}</p>
              </div>
              <div className="p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Last Active</p>
                <p className="text-lg font-semibold">{selectedUser.lastActive}</p>
              </div>
            </div>

            <div className="flex gap-2">
              <button
                className="flex-1 py-2 rounded-lg text-sm font-medium"
                style={{ background: 'var(--bg-input)', color: 'var(--text-primary)', border: '1px solid var(--border)' }}
              >
                Change Role
              </button>
              {selectedUser.status === 'suspended' ? (
                <button
                  className="flex-1 py-2 rounded-lg text-sm font-medium"
                  style={{ background: 'var(--green-muted)', color: 'var(--green)', border: '1px solid var(--green)' }}
                >
                  Unsuspend
                </button>
              ) : (
                <button
                  className="flex-1 py-2 rounded-lg text-sm font-medium"
                  style={{ background: 'var(--red-muted)', color: 'var(--red)', border: '1px solid var(--red)' }}
                >
                  Suspend
                </button>
              )}
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}
